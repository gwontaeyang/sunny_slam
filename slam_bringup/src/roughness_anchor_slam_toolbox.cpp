#include "slam_bringup/roughness_anchor_slam_toolbox.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace
{

std::string trim(const std::string & value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> splitCsvLine(const std::string & line)
{
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) {
    fields.push_back(trim(field));
  }
  if (!line.empty() && line.back() == ',') {
    fields.emplace_back();
  }
  return fields;
}

double finiteDouble(
  const std::unordered_map<std::string, std::string> & row,
  const std::string & key,
  const double fallback = std::numeric_limits<double>::quiet_NaN())
{
  const auto it = row.find(key);
  if (it == row.end()) {
    return fallback;
  }
  try {
    const double out = std::stod(it->second);
    return std::isfinite(out) ? out : fallback;
  } catch (const std::exception &) {
    return fallback;
  }
}

int finiteInt(
  const std::unordered_map<std::string, std::string> & row,
  const std::string & key,
  const int fallback = -1)
{
  const double value = finiteDouble(row, key, static_cast<double>(fallback));
  return std::isfinite(value) ? static_cast<int>(value) : fallback;
}

std::string stringValue(
  const std::unordered_map<std::string, std::string> & row,
  const std::string & key)
{
  const auto it = row.find(key);
  return it == row.end() ? "" : it->second;
}

double normalizedAngle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

karto::Pose2 relativePose(const karto::Pose2 & from, const karto::Pose2 & to)
{
  karto::Transform transform(from, karto::Pose2());
  return transform.TransformPose(to);
}

karto::Pose2 composePose(const karto::Pose2 & first, const karto::Pose2 & second)
{
  const double cosine = std::cos(first.GetHeading());
  const double sine = std::sin(first.GetHeading());
  return karto::Pose2(
    first.GetX() + cosine * second.GetX() - sine * second.GetY(),
    first.GetY() + sine * second.GetX() + cosine * second.GetY(),
    normalizedAngle(first.GetHeading() + second.GetHeading()));
}

karto::Pose2 inversePose(const karto::Pose2 & pose)
{
  const double cosine = std::cos(pose.GetHeading());
  const double sine = std::sin(pose.GetHeading());
  return karto::Pose2(
    -cosine * pose.GetX() - sine * pose.GetY(),
    sine * pose.GetX() - cosine * pose.GetY(),
    normalizedAngle(-pose.GetHeading()));
}

double poseTranslationDistance(const karto::Pose2 & first, const karto::Pose2 & second)
{
  return std::hypot(first.GetX() - second.GetX(), first.GetY() - second.GetY());
}

std::string normalizedMode(std::string value)
{
  std::transform(
    value.begin(), value.end(), value.begin(),
    [](unsigned char character) {return static_cast<char>(std::tolower(character));});
  return value;
}

enum class RawWindowSide
{
  kCentered,
  kBefore,
  kAfter,
};

RawWindowSide rawWindowSide(const std::string & role, bool target)
{
  const std::string token = target ? role.substr(0, role.find("__")) :
    role.substr(
    role.find("__") == std::string::npos ? 0 : role.find("__") + 2);
  if (token.find("_pre") != std::string::npos) {
    return RawWindowSide::kBefore;
  }
  if (token.find("_post") != std::string::npos) {
    return RawWindowSide::kAfter;
  }
  return RawWindowSide::kCentered;
}

}  // namespace

namespace slam_bringup
{

RoughnessAnchorSlamToolbox::RoughnessAnchorSlamToolbox(rclcpp::NodeOptions options)
: slam_toolbox::SynchronousSlamToolbox(options)
{
  declare_parameter("enable_roughness_anchor_constraints", false);
  declare_parameter("roughness_anchor_inject_constraints", true);
  declare_parameter("roughness_anchor_replace_existing_edge", false);
  declare_parameter("roughness_anchor_constraints_csv", "");
  declare_parameter("roughness_anchor_measurement_mode", "scan_match");
  declare_parameter("roughness_anchor_event_a", -1);
  declare_parameter("roughness_anchor_event_b", -1);
  declare_parameter("roughness_anchor_max_stamp_dt_s", 8.0);
  declare_parameter("roughness_anchor_min_scan_iou", 0.15);
  declare_parameter("roughness_anchor_max_scan_nn_p50_m", 0.10);
  declare_parameter("roughness_anchor_max_abs_scan_yaw_deg", 5.0);
  declare_parameter("roughness_anchor_max_input_odom_gap_m", 0.10);
  declare_parameter("roughness_anchor_min_pair_residual_m", 0.05);
  declare_parameter("roughness_anchor_max_vertex_odom_gap_m", 0.50);
  declare_parameter("roughness_anchor_max_graph_odom_gap_m", 0.05);
  declare_parameter("roughness_anchor_scan_match_max_graph_odom_gap_m", 0.75);
  declare_parameter("roughness_anchor_scan_match_use_cached_anchor_scans", true);
  declare_parameter("roughness_anchor_scan_match_require_cached_anchor_scans", true);
  declare_parameter("roughness_anchor_scan_match_raw_scan_max_dt_s", 0.10);
  declare_parameter("roughness_anchor_scan_match_raw_window_half_duration_s", 0.0);
  declare_parameter("roughness_anchor_scan_match_raw_window_min_scans", 3);
  declare_parameter("roughness_anchor_scan_match_raw_window_max_scans", 9);
  declare_parameter("roughness_anchor_scan_match_raw_window_min_separation_s", 0.04);
  declare_parameter("roughness_anchor_scan_match_raw_window_stable_side_only", false);
  declare_parameter("roughness_anchor_scan_match_base_half_window", 0);
  declare_parameter("roughness_anchor_scan_match_max_base_dt_s", 8.0);
  declare_parameter("roughness_anchor_scan_match_min_coarse_response", 0.70);
  declare_parameter("roughness_anchor_scan_match_min_fine_response", 0.55);
  declare_parameter("roughness_anchor_scan_match_max_coarse_variance", 0.16);
  declare_parameter("roughness_anchor_scan_match_max_translation_innovation_m", 0.75);
  declare_parameter("roughness_anchor_scan_match_max_yaw_innovation_deg", 10.0);
  declare_parameter("roughness_anchor_scan_match_require_bidirectional", true);
  declare_parameter("roughness_anchor_scan_match_max_bidirectional_translation_error_m", 0.15);
  declare_parameter("roughness_anchor_scan_match_max_bidirectional_yaw_error_deg", 3.0);
  declare_parameter("roughness_anchor_scan_match_covariance_inflation", 2.0);
  declare_parameter("roughness_anchor_scan_match_min_translation_sigma_m", 0.05);
  declare_parameter("roughness_anchor_scan_match_min_yaw_sigma_deg", 2.0);
  declare_parameter("roughness_anchor_translation_sigma_m", 0.10);
  declare_parameter("roughness_anchor_yaw_sigma_rad", 1.0);
  declare_parameter("roughness_anchor_min_state_id_separation", 5);
  declare_parameter("roughness_anchor_max_constraint_pairs", 8);
  declare_parameter("roughness_anchor_require_complete_pair", true);
  declare_parameter("roughness_anchor_update_map_after_constraints", true);
}

void RoughnessAnchorSlamToolbox::configure()
{
  slam_toolbox::SynchronousSlamToolbox::configure();
  loadRoughnessAnchorParameters();
  if (enable_constraints_ && constraints_loaded_) {
    anchor_timer_ = create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        tryInjectPendingConstraints();
      });
  }
}

void RoughnessAnchorSlamToolbox::laserCallback(
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
{
  slam_toolbox::SynchronousSlamToolbox::laserCallback(scan);
  cacheConstraintScan(scan);
}

void RoughnessAnchorSlamToolbox::loadRoughnessAnchorParameters()
{
  get_parameter("enable_roughness_anchor_constraints", enable_constraints_);
  get_parameter("roughness_anchor_inject_constraints", inject_constraints_);
  get_parameter("roughness_anchor_replace_existing_edge", replace_existing_edge_);
  get_parameter("roughness_anchor_constraints_csv", constraints_csv_);
  get_parameter("roughness_anchor_measurement_mode", measurement_mode_);
  get_parameter("roughness_anchor_event_a", filter_event_a_);
  get_parameter("roughness_anchor_event_b", filter_event_b_);
  get_parameter("roughness_anchor_max_stamp_dt_s", max_stamp_dt_s_);
  get_parameter("roughness_anchor_min_scan_iou", min_scan_iou_);
  get_parameter("roughness_anchor_max_scan_nn_p50_m", max_scan_nn_p50_m_);
  get_parameter("roughness_anchor_max_abs_scan_yaw_deg", max_abs_scan_yaw_deg_);
  get_parameter("roughness_anchor_max_input_odom_gap_m", max_input_odom_gap_m_);
  get_parameter("roughness_anchor_min_pair_residual_m", min_pair_residual_m_);
  get_parameter("roughness_anchor_max_vertex_odom_gap_m", max_vertex_odom_gap_m_);
  get_parameter("roughness_anchor_max_graph_odom_gap_m", max_graph_odom_gap_m_);
  get_parameter(
    "roughness_anchor_scan_match_max_graph_odom_gap_m",
    scan_match_max_graph_odom_gap_m_);
  get_parameter(
    "roughness_anchor_scan_match_use_cached_anchor_scans",
    scan_match_use_cached_anchor_scans_);
  get_parameter(
    "roughness_anchor_scan_match_require_cached_anchor_scans",
    scan_match_require_cached_anchor_scans_);
  get_parameter(
    "roughness_anchor_scan_match_raw_scan_max_dt_s",
    scan_match_raw_scan_max_dt_s_);
  get_parameter(
    "roughness_anchor_scan_match_raw_window_half_duration_s",
    scan_match_raw_window_half_duration_s_);
  get_parameter(
    "roughness_anchor_scan_match_raw_window_min_scans",
    scan_match_raw_window_min_scans_);
  get_parameter(
    "roughness_anchor_scan_match_raw_window_max_scans",
    scan_match_raw_window_max_scans_);
  get_parameter(
    "roughness_anchor_scan_match_raw_window_min_separation_s",
    scan_match_raw_window_min_separation_s_);
  get_parameter(
    "roughness_anchor_scan_match_raw_window_stable_side_only",
    scan_match_raw_window_stable_side_only_);
  get_parameter(
    "roughness_anchor_scan_match_base_half_window",
    scan_match_base_half_window_);
  get_parameter(
    "roughness_anchor_scan_match_max_base_dt_s",
    scan_match_max_base_dt_s_);
  get_parameter(
    "roughness_anchor_scan_match_min_coarse_response",
    scan_match_min_coarse_response_);
  get_parameter(
    "roughness_anchor_scan_match_min_fine_response",
    scan_match_min_fine_response_);
  get_parameter(
    "roughness_anchor_scan_match_max_coarse_variance",
    scan_match_max_coarse_variance_);
  get_parameter(
    "roughness_anchor_scan_match_max_translation_innovation_m",
    scan_match_max_translation_innovation_m_);
  double scan_match_max_yaw_innovation_deg = 10.0;
  get_parameter(
    "roughness_anchor_scan_match_max_yaw_innovation_deg",
    scan_match_max_yaw_innovation_deg);
  get_parameter(
    "roughness_anchor_scan_match_require_bidirectional",
    scan_match_require_bidirectional_);
  get_parameter(
    "roughness_anchor_scan_match_max_bidirectional_translation_error_m",
    scan_match_max_bidirectional_translation_error_m_);
  double scan_match_max_bidirectional_yaw_error_deg = 3.0;
  get_parameter(
    "roughness_anchor_scan_match_max_bidirectional_yaw_error_deg",
    scan_match_max_bidirectional_yaw_error_deg);
  get_parameter(
    "roughness_anchor_scan_match_covariance_inflation",
    scan_match_covariance_inflation_);
  get_parameter(
    "roughness_anchor_scan_match_min_translation_sigma_m",
    scan_match_min_translation_sigma_m_);
  double scan_match_min_yaw_sigma_deg = 2.0;
  get_parameter(
    "roughness_anchor_scan_match_min_yaw_sigma_deg",
    scan_match_min_yaw_sigma_deg);
  get_parameter("roughness_anchor_translation_sigma_m", translation_sigma_m_);
  get_parameter("roughness_anchor_yaw_sigma_rad", yaw_sigma_rad_);
  get_parameter("roughness_anchor_min_state_id_separation", min_state_id_separation_);
  get_parameter("roughness_anchor_max_constraint_pairs", max_constraint_pairs_);
  get_parameter("roughness_anchor_require_complete_pair", require_complete_anchor_pair_);
  get_parameter("roughness_anchor_update_map_after_constraints", update_map_after_constraints_);

  measurement_mode_ = normalizedMode(measurement_mode_);
  if (measurement_mode_ != "scan_match" && measurement_mode_ != "odom") {
    RCLCPP_WARN(
      get_logger(),
      "Unknown roughness_anchor_measurement_mode '%s'; using scan_match.",
      measurement_mode_.c_str());
    measurement_mode_ = "scan_match";
  }
  scan_match_max_yaw_innovation_rad_ =
    std::max(scan_match_max_yaw_innovation_deg, 0.0) * M_PI / 180.0;
  scan_match_max_bidirectional_yaw_error_rad_ =
    std::max(scan_match_max_bidirectional_yaw_error_deg, 0.0) * M_PI / 180.0;
  scan_match_min_yaw_sigma_rad_ =
    std::max(scan_match_min_yaw_sigma_deg, 1.0e-3) * M_PI / 180.0;
  translation_sigma_m_ = std::max(translation_sigma_m_, 1.0e-3);
  yaw_sigma_rad_ = std::max(yaw_sigma_rad_, 1.0e-3);
  max_stamp_dt_s_ = std::max(max_stamp_dt_s_, 1.0e-3);
  max_input_odom_gap_m_ = std::max(max_input_odom_gap_m_, 0.0);
  min_pair_residual_m_ = std::max(min_pair_residual_m_, 0.0);
  max_vertex_odom_gap_m_ = std::max(max_vertex_odom_gap_m_, 1.0e-3);
  max_graph_odom_gap_m_ = std::max(max_graph_odom_gap_m_, 1.0e-3);
  scan_match_max_graph_odom_gap_m_ =
    std::max(scan_match_max_graph_odom_gap_m_, 1.0e-3);
  scan_match_raw_scan_max_dt_s_ =
    std::max(scan_match_raw_scan_max_dt_s_, 1.0e-3);
  scan_match_raw_window_half_duration_s_ =
    std::max(scan_match_raw_window_half_duration_s_, 0.0);
  scan_match_raw_window_min_scans_ =
    std::max(scan_match_raw_window_min_scans_, 1);
  scan_match_raw_window_max_scans_ =
    std::max(scan_match_raw_window_max_scans_, scan_match_raw_window_min_scans_);
  scan_match_raw_window_min_separation_s_ =
    std::max(scan_match_raw_window_min_separation_s_, 0.0);
  scan_match_base_half_window_ = std::max(scan_match_base_half_window_, 0);
  scan_match_max_base_dt_s_ = std::max(scan_match_max_base_dt_s_, 1.0e-3);
  scan_match_min_coarse_response_ =
    std::clamp(scan_match_min_coarse_response_, 0.0, 1.0);
  scan_match_min_fine_response_ =
    std::clamp(scan_match_min_fine_response_, 0.0, 1.0);
  scan_match_max_coarse_variance_ =
    std::max(scan_match_max_coarse_variance_, 1.0e-6);
  scan_match_max_translation_innovation_m_ =
    std::max(scan_match_max_translation_innovation_m_, 0.0);
  scan_match_max_bidirectional_translation_error_m_ =
    std::max(scan_match_max_bidirectional_translation_error_m_, 0.0);
  scan_match_covariance_inflation_ =
    std::max(scan_match_covariance_inflation_, 1.0);
  scan_match_min_translation_sigma_m_ =
    std::max(scan_match_min_translation_sigma_m_, 1.0e-3);
  min_state_id_separation_ = std::max(min_state_id_separation_, 0);
  max_constraint_pairs_ = std::max(max_constraint_pairs_, 0);

  if (!enable_constraints_) {
    RCLCPP_INFO(get_logger(), "Roughness-anchor graph constraints are disabled.");
    return;
  }
  if (constraints_csv_.empty()) {
    RCLCPP_WARN(
      get_logger(),
      "Roughness-anchor graph constraints are enabled but roughness_anchor_constraints_csv is empty.");
    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "Roughness-anchor measurement mode=%s inject_constraints=%s replace_existing=%s "
    "scan_match_coarse/fine=%.3f/%.3f bidirectional=%s cached_raw=%s/%s "
    "raw_max_dt=%.3f raw_window=%.3fs/%d-%d stable_side=%s base_half_window=%d.",
    measurement_mode_.c_str(),
    inject_constraints_ ? "true" : "false",
    replace_existing_edge_ ? "true" : "false",
    scan_match_min_coarse_response_,
    scan_match_min_fine_response_,
    scan_match_require_bidirectional_ ? "true" : "false",
    scan_match_use_cached_anchor_scans_ ? "enabled" : "disabled",
    scan_match_require_cached_anchor_scans_ ? "required" : "optional",
    scan_match_raw_scan_max_dt_s_,
    scan_match_raw_window_half_duration_s_,
    scan_match_raw_window_min_scans_,
    scan_match_raw_window_max_scans_,
    scan_match_raw_window_stable_side_only_ ? "true" : "false",
    scan_match_base_half_window_);
  loadConstraintsFromCsv(constraints_csv_);
}

void RoughnessAnchorSlamToolbox::loadConstraintsFromCsv(const std::string & csv_path)
{
  constraints_.clear();
  injected_edges_.clear();
  constraints_loaded_ = false;

  std::ifstream input(csv_path);
  if (!input.is_open()) {
    RCLCPP_ERROR(get_logger(), "Failed to open roughness-anchor CSV: %s", csv_path.c_str());
    return;
  }

  std::string line;
  if (!std::getline(input, line)) {
    RCLCPP_ERROR(get_logger(), "Roughness-anchor CSV is empty: %s", csv_path.c_str());
    return;
  }

  const std::vector<std::string> headers = splitCsvLine(line);
  std::size_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (trim(line).empty()) {
      continue;
    }
    const std::vector<std::string> values = splitCsvLine(line);
    std::unordered_map<std::string, std::string> row;
    for (std::size_t i = 0; i < headers.size() && i < values.size(); ++i) {
      row[headers[i]] = values[i];
    }

    const int event_a = finiteInt(row, "event_a");
    const int event_b = finiteInt(row, "event_b");
    if (filter_event_a_ >= 0 && event_a != filter_event_a_) {
      continue;
    }
    if (filter_event_b_ >= 0 && event_b != filter_event_b_) {
      continue;
    }

    const double iou = finiteDouble(row, "centered_best_yaw_iou");
    const double nn_p50_m = finiteDouble(row, "odom_frame_nn_p50_m");
    const double yaw_deg = finiteDouble(row, "best_yaw_correction_deg");
    if (!std::isfinite(iou) || iou < min_scan_iou_) {
      continue;
    }
    if (!std::isfinite(nn_p50_m) || nn_p50_m > max_scan_nn_p50_m_) {
      continue;
    }
    if (!std::isfinite(yaw_deg) || std::abs(yaw_deg) > max_abs_scan_yaw_deg_) {
      continue;
    }

    const double a_stamp_s = finiteDouble(row, "a_stamp_s");
    const double b_stamp_s = finiteDouble(row, "b_stamp_s");
    if (!std::isfinite(a_stamp_s) || !std::isfinite(b_stamp_s)) {
      RCLCPP_WARN(
        get_logger(),
        "Skipping roughness-anchor CSV line %zu with invalid stamps.",
        line_number);
      continue;
    }

    RoughnessAnchorConstraint constraint;
    constraint.event_a = event_a;
    constraint.event_b = event_b;
    constraint.role = stringValue(row, "anchor_role");
    constraint.target_stamp_s = a_stamp_s;
    constraint.source_stamp_s = b_stamp_s;
    constraint.scan_iou = iou;
    constraint.scan_nn_p50_m = nn_p50_m;
    constraint.scan_yaw_deg = yaw_deg;
    constraint.input_odom_gap_m = finiteDouble(row, "input_anchor_odom_gap_m");
    constraint.input_residual_m = finiteDouble(row, "input_anchor_residual_m");
    constraint.pair_residual_m = finiteDouble(row, "pair_anchor_residual_m");
    if (
      !std::isfinite(constraint.input_odom_gap_m) ||
      constraint.input_odom_gap_m > max_input_odom_gap_m_)
    {
      continue;
    }
    if (
      !std::isfinite(constraint.pair_residual_m) ||
      constraint.pair_residual_m < min_pair_residual_m_)
    {
      continue;
    }
    constraint.target_odom_x_m = finiteDouble(row, "a_odom_x_m");
    constraint.target_odom_y_m = finiteDouble(row, "a_odom_y_m");
    constraint.target_odom_yaw_rad = finiteDouble(row, "a_odom_yaw_rad");
    constraint.source_odom_x_m = finiteDouble(row, "b_odom_x_m");
    constraint.source_odom_y_m = finiteDouble(row, "b_odom_y_m");
    constraint.source_odom_yaw_rad = finiteDouble(row, "b_odom_yaw_rad");
    constraints_.push_back(constraint);
  }

  if (require_complete_anchor_pair_) {
    std::map<std::pair<int, int>, std::set<std::string>> roles_by_pair;
    for (const auto & constraint : constraints_) {
      roles_by_pair[{constraint.event_a, constraint.event_b}].insert(constraint.role);
    }
    std::vector<RoughnessAnchorConstraint> complete_constraints;
    for (const auto & constraint : constraints_) {
      const auto key = std::make_pair(constraint.event_a, constraint.event_b);
      if (roles_by_pair[key].size() >= 2) {
        complete_constraints.push_back(constraint);
      }
    }
    constraints_.swap(complete_constraints);
  }

  if (max_constraint_pairs_ > 0) {
    std::set<std::pair<int, int>> selected_pairs;
    std::vector<RoughnessAnchorConstraint> limited_constraints;
    for (const auto & constraint : constraints_) {
      const auto key = std::make_pair(constraint.event_a, constraint.event_b);
      if (
        selected_pairs.find(key) == selected_pairs.end() &&
        static_cast<int>(selected_pairs.size()) >= max_constraint_pairs_)
      {
        continue;
      }
      selected_pairs.insert(key);
      limited_constraints.push_back(constraint);
    }
    constraints_.swap(limited_constraints);
  }

  constraints_loaded_ = !constraints_.empty();
  RCLCPP_INFO(
    get_logger(),
    "Loaded %zu roughness-anchor graph constraints from %s.",
    constraints_.size(),
    csv_path.c_str());
}

void RoughnessAnchorSlamToolbox::cacheConstraintScan(
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
{
  if (
    !scan_match_use_cached_anchor_scans_ || !constraints_loaded_ || !scan)
  {
    return;
  }

  const double stamp_s = rclcpp::Time(scan->header.stamp).seconds();
  latest_raw_scan_stamp_s_ = std::max(latest_raw_scan_stamp_s_, stamp_s);
  karto::Pose2 scan_odom_pose;
  const bool have_scan_odom =
    scan_match_raw_window_half_duration_s_ > 0.0 &&
    pose_helper_ &&
    pose_helper_->getOdomPose(scan_odom_pose, rclcpp::Time(scan->header.stamp));

  for (auto & constraint : constraints_) {
    const auto cache_window_scan =
      [this, &scan, stamp_s, &scan_odom_pose, have_scan_odom](
      double anchor_stamp_s,
      RawWindowSide side,
      std::vector<CachedRoughnessAnchorScan> & window)
      {
        const double relative_stamp_s = stamp_s - anchor_stamp_s;
        bool inside_window =
          std::abs(relative_stamp_s) <= scan_match_raw_window_half_duration_s_;
        if (scan_match_raw_window_stable_side_only_) {
          if (side == RawWindowSide::kBefore) {
            inside_window =
              relative_stamp_s >= -scan_match_raw_window_half_duration_s_ &&
              relative_stamp_s <= scan_match_raw_scan_max_dt_s_;
          } else if (side == RawWindowSide::kAfter) {
            inside_window =
              relative_stamp_s >= -scan_match_raw_scan_max_dt_s_ &&
              relative_stamp_s <= scan_match_raw_window_half_duration_s_;
          }
        }
        if (
          !have_scan_odom ||
          !inside_window)
        {
          return;
        }

        for (auto & cached : window) {
          if (
            std::abs(cached.stamp_s - stamp_s) <
            scan_match_raw_window_min_separation_s_)
          {
            if (
              std::abs(stamp_s - anchor_stamp_s) <
              std::abs(cached.stamp_s - anchor_stamp_s))
            {
              cached = CachedRoughnessAnchorScan{scan, scan_odom_pose, stamp_s};
            }
            return;
          }
        }

        window.push_back(CachedRoughnessAnchorScan{scan, scan_odom_pose, stamp_s});
        if (
          static_cast<int>(window.size()) >
          scan_match_raw_window_max_scans_)
        {
          const auto farthest = std::max_element(
            window.begin(), window.end(),
            [anchor_stamp_s](const auto & first, const auto & second) {
              return std::abs(first.stamp_s - anchor_stamp_s) <
              std::abs(second.stamp_s - anchor_stamp_s);
            });
          window.erase(farthest);
        }
        std::sort(
          window.begin(), window.end(),
          [](const auto & first, const auto & second) {
            return first.stamp_s < second.stamp_s;
          });
      };

    cache_window_scan(
      constraint.target_stamp_s,
      rawWindowSide(constraint.role, true),
      constraint.target_raw_window);
    cache_window_scan(
      constraint.source_stamp_s,
      rawWindowSide(constraint.role, false),
      constraint.source_raw_window);

    const double target_dt_s = std::abs(stamp_s - constraint.target_stamp_s);
    if (
      target_dt_s <= scan_match_raw_scan_max_dt_s_ &&
      target_dt_s < constraint.target_raw_scan_dt_s)
    {
      constraint.target_raw_scan = scan;
      constraint.target_raw_scan_dt_s = target_dt_s;
      RCLCPP_INFO(
        get_logger(),
        "Cached roughness-anchor target raw scan event %d->%d role=%s "
        "scan_stamp=%.6f target_stamp=%.6f dt=%.6f s.",
        constraint.event_a,
        constraint.event_b,
        constraint.role.c_str(),
        stamp_s,
        constraint.target_stamp_s,
        target_dt_s);
    }

    const double source_dt_s = std::abs(stamp_s - constraint.source_stamp_s);
    if (
      source_dt_s <= scan_match_raw_scan_max_dt_s_ &&
      source_dt_s < constraint.source_raw_scan_dt_s)
    {
      constraint.source_raw_scan = scan;
      constraint.source_raw_scan_dt_s = source_dt_s;
      RCLCPP_INFO(
        get_logger(),
        "Cached roughness-anchor source raw scan event %d->%d role=%s "
        "scan_stamp=%.6f source_stamp=%.6f dt=%.6f s.",
        constraint.event_a,
        constraint.event_b,
        constraint.role.c_str(),
        stamp_s,
        constraint.source_stamp_s,
        source_dt_s);
    }
  }
}

void RoughnessAnchorSlamToolbox::tryInjectPendingConstraints()
{
  if (!enable_constraints_) {
    return;
  }
  if (!constraints_loaded_) {
    if (!load_error_reported_) {
      RCLCPP_WARN(
        get_logger(),
        "Roughness-anchor constraints are enabled but no usable constraints were loaded.");
      load_error_reported_ = true;
    }
    return;
  }

  bool injected_any = false;
  std::size_t pending_count = 0;
  {
    // Serialize graph mutations with synchronous scan processing. updateMap()
    // acquires this mutex itself, so it must be called after this scope.
    boost::mutex::scoped_lock lock(smapper_mutex_);
    for (auto & constraint : constraints_) {
      if (constraint.injected) {
        continue;
      }
      ++pending_count;
      injected_any = tryInjectConstraint(constraint) || injected_any;
    }

    if (!injected_any && pending_count > 0) {
      ++anchor_timer_tick_;
      if (anchor_timer_tick_ == 1 || anchor_timer_tick_ % 10 == 0) {
        logPendingConstraintStatus();
      }
    }
  }

  if (injected_any && update_map_after_constraints_) {
    updateMap();
  }
  if (pending_count == 0 && anchor_timer_) {
    anchor_timer_->cancel();
  }
}

bool RoughnessAnchorSlamToolbox::tryInjectConstraint(RoughnessAnchorConstraint & constraint)
{
  if (!smapper_ || !smapper_->getMapper() || !smapper_->getMapper()->GetGraph()) {
    return false;
  }
  const auto window_completion_stamp =
    [this](double anchor_stamp_s, RawWindowSide side) {
      if (
        scan_match_raw_window_stable_side_only_ &&
        side == RawWindowSide::kBefore)
      {
        return anchor_stamp_s;
      }
      return anchor_stamp_s + scan_match_raw_window_half_duration_s_;
    };
  if (
    scan_match_raw_window_half_duration_s_ > 0.0 &&
    latest_raw_scan_stamp_s_ <
    std::max(
      window_completion_stamp(
        constraint.target_stamp_s, rawWindowSide(constraint.role, true)),
      window_completion_stamp(
        constraint.source_stamp_s, rawWindowSide(constraint.role, false))))
  {
    return false;
  }

  VertexMatch source_match = findNearestVertex(
    constraint.source_stamp_s,
    constraint.source_odom_x_m,
    constraint.source_odom_y_m);
  VertexMatch target_match = findNearestVertex(
    constraint.target_stamp_s,
    constraint.target_odom_x_m,
    constraint.target_odom_y_m);
  if (!source_match.vertex || !target_match.vertex) {
    return false;
  }
  const bool source_has_odom_target =
    std::isfinite(constraint.source_odom_x_m) && std::isfinite(constraint.source_odom_y_m);
  const bool target_has_odom_target =
    std::isfinite(constraint.target_odom_x_m) && std::isfinite(constraint.target_odom_y_m);
  if (source_match.dt_s > max_stamp_dt_s_) {
    return false;
  }
  if (target_match.dt_s > max_stamp_dt_s_) {
    return false;
  }
  if (
    source_has_odom_target && source_match.odom_gap_m > max_vertex_odom_gap_m_)
  {
    return false;
  }
  if (
    target_has_odom_target && target_match.odom_gap_m > max_vertex_odom_gap_m_)
  {
    return false;
  }

  auto * source_scan = source_match.vertex->GetObject();
  auto * target_scan = target_match.vertex->GetObject();
  if (!source_scan || !target_scan) {
    return false;
  }

  const int source_id = source_scan->GetStateId();
  const int target_id = target_scan->GetStateId();
  if (source_id == target_id) {
    return false;
  }
  if (std::abs(source_id - target_id) < min_state_id_separation_) {
    return false;
  }

  const auto edge_key = std::make_pair(
    std::min(source_id, target_id), std::max(
      source_id,
      target_id));
  if (injected_edges_.find(edge_key) != injected_edges_.end()) {
    constraint.injected = true;
    return false;
  }
  auto * solver = smapper_->getMapper()->getScanSolver();
  if (!solver) {
    RCLCPP_ERROR(
      get_logger(),
      "Failed to inject roughness-anchor edge event %d->%d role=%s because scan solver is null.",
      constraint.event_a,
      constraint.event_b,
      constraint.role.c_str());
    return false;
  }

  auto * from_scan = target_scan;
  auto * to_scan = source_scan;
  if (source_scan->GetTime() < target_scan->GetTime()) {
    from_scan = source_scan;
    to_scan = target_scan;
  }

  const karto::Pose2 from_odom = from_scan->GetOdometricPose();
  const karto::Pose2 to_odom = to_scan->GetOdometricPose();
  const karto::Pose2 from_corrected = from_scan->GetCorrectedPose();
  const double graph_odom_dx = to_odom.GetX() - from_odom.GetX();
  const double graph_odom_dy = to_odom.GetY() - from_odom.GetY();
  const double graph_odom_gap_m = std::hypot(graph_odom_dx, graph_odom_dy);
  const double graph_odom_gate_m =
    measurement_mode_ == "scan_match" ?
    scan_match_max_graph_odom_gap_m_ : max_graph_odom_gap_m_;
  if (graph_odom_gap_m > graph_odom_gate_m) {
    const auto reject_ids = std::make_pair(source_id, target_id);
    if (constraint.last_graph_odom_reject_ids != reject_ids) {
      RCLCPP_WARN(
        get_logger(),
        "Deferring roughness-anchor graph edge event %d->%d role=%s because graph odom gap "
        "%.3f m exceeds %.3f m. source_id=%d target_id=%d from_id=%d to_id=%d "
        "input_odom_gap=%.3f mode=%s; a later graph vertex may satisfy the gate.",
        constraint.event_a,
        constraint.event_b,
        constraint.role.c_str(),
        graph_odom_gap_m,
        graph_odom_gate_m,
        source_id,
        target_id,
        from_scan->GetStateId(),
        to_scan->GetStateId(),
        constraint.input_odom_gap_m,
        measurement_mode_.c_str());
      constraint.last_graph_odom_reject_ids = reject_ids;
    }
    return false;
  }

  karto::Pose2 to_expected;
  karto::Matrix3 edge_covariance;
  ScanMatchResult forward_match;
  ScanMatchResult reverse_match;
  if (measurement_mode_ == "scan_match") {
    if (constraint.last_scan_match_reject_ids == edge_key) {
      return false;
    }
    std::string rejection_reason;
    if (!makeScanMatchMeasurement(
        constraint,
        from_scan,
        to_scan,
        to_expected,
        edge_covariance,
        forward_match,
        reverse_match,
        rejection_reason))
    {
      if (rejection_reason != "missing_cached_anchor_scan") {
        constraint.last_scan_match_reject_ids = edge_key;
      }
      RCLCPP_WARN(
        get_logger(),
        "Rejected roughness-anchor scan match event %d->%d role=%s from_id=%d to_id=%d "
        "reason=%s coarse_response=%.4f fine_response=%.4f coarse_var_xy=%.6f/%.6f "
        "innovation_xy=%.4f innovation_yaw_deg=%.3f base_scans=%zu "
        "reverse_coarse=%.4f reverse_fine=%.4f reverse_var_xy=%.6f/%.6f "
        "reverse_innovation_xy=%.4f reverse_innovation_yaw_deg=%.3f "
        "reverse_base_scans=%zu bidirectional_xy=%.4f bidirectional_yaw_deg=%.3f "
        "raw_window_scans=%zu/%zu.",
        constraint.event_a,
        constraint.event_b,
        constraint.role.c_str(),
        from_scan->GetStateId(),
        to_scan->GetStateId(),
        rejection_reason.c_str(),
        forward_match.coarse_response,
        forward_match.fine_response,
        forward_match.coarse_variance_x,
        forward_match.coarse_variance_y,
        forward_match.translation_innovation_m,
        forward_match.yaw_innovation_rad * 180.0 / M_PI,
        forward_match.base_scan_count,
        reverse_match.coarse_response,
        reverse_match.fine_response,
        reverse_match.coarse_variance_x,
        reverse_match.coarse_variance_y,
        reverse_match.translation_innovation_m,
        reverse_match.yaw_innovation_rad * 180.0 / M_PI,
        reverse_match.base_scan_count,
        forward_match.bidirectional_translation_error_m,
        forward_match.bidirectional_yaw_error_rad * 180.0 / M_PI,
        constraint.target_raw_window.size(),
        constraint.source_raw_window.size());
      return false;
    }
    RCLCPP_INFO(
      get_logger(),
      "Accepted roughness-anchor scan match event %d->%d role=%s from_id=%d to_id=%d "
      "coarse_response=%.4f fine_response=%.4f coarse_var_xy=%.6f/%.6f "
      "innovation_xy=%.4f innovation_yaw_deg=%.3f reverse_coarse=%.4f "
      "reverse_fine=%.4f base_scans=%zu/%zu bidirectional_xy=%.4f "
      "bidirectional_yaw_deg=%.3f raw_scan_dt=%.4f/%.4f.",
      constraint.event_a,
      constraint.event_b,
      constraint.role.c_str(),
      from_scan->GetStateId(),
      to_scan->GetStateId(),
      forward_match.coarse_response,
      forward_match.fine_response,
      forward_match.coarse_variance_x,
      forward_match.coarse_variance_y,
      forward_match.translation_innovation_m,
      forward_match.yaw_innovation_rad * 180.0 / M_PI,
      reverse_match.coarse_response,
      reverse_match.fine_response,
      forward_match.base_scan_count,
      reverse_match.base_scan_count,
      forward_match.bidirectional_translation_error_m,
      forward_match.bidirectional_yaw_error_rad * 180.0 / M_PI,
      constraint.target_raw_scan_dt_s,
      constraint.source_raw_scan_dt_s);
  } else {
    karto::Transform odom_to_corrected(from_odom, from_corrected);
    to_expected = odom_to_corrected.TransformPose(to_odom);
    const double yaw_delta = normalizedAngle(
      to_expected.GetHeading() - to_scan->GetCorrectedPose().GetHeading());
    if (std::abs(yaw_delta) > yaw_sigma_rad_ * 3.0) {
      to_expected.SetHeading(to_scan->GetCorrectedPose().GetHeading());
    }
    edge_covariance = makeAnchorCovariance();
  }

  if (!inject_constraints_) {
    constraint.injected = true;
    RCLCPP_INFO(
      get_logger(),
      "Validated roughness-anchor graph measurement without edge injection event %d->%d "
      "role=%s from_id=%d to_id=%d graph_odom_gap=%.3f measurement_mode=%s.",
      constraint.event_a,
      constraint.event_b,
      constraint.role.c_str(),
      from_scan->GetStateId(),
      to_scan->GetStateId(),
      graph_odom_gap_m,
      measurement_mode_.c_str());
    return false;
  }

  kt_bool is_new_edge = false;
  auto * edge = smapper_->getMapper()->GetGraph()->AddEdge(from_scan, to_scan, is_new_edge);
  if (!edge) {
    RCLCPP_ERROR(
      get_logger(),
      "Failed to create or find roughness-anchor edge event %d->%d role=%s.",
      constraint.event_a,
      constraint.event_b,
      constraint.role.c_str());
    return false;
  }

  bool replaced_existing_edge = false;
  karto::Pose2 old_edge_measurement;
  karto::Matrix3 old_edge_covariance;
  if (!is_new_edge) {
    if (!replace_existing_edge_) {
      constraint.injected = true;
      RCLCPP_WARN(
        get_logger(),
        "Skipped roughness-anchor edge event %d->%d role=%s because edge %d<->%d already exists.",
        constraint.event_a,
        constraint.event_b,
        constraint.role.c_str(),
        source_id,
        target_id);
      return false;
    }
    if (
      edge->GetSource()->GetObject() != from_scan ||
      edge->GetTarget()->GetObject() != to_scan)
    {
      constraint.injected = true;
      RCLCPP_WARN(
        get_logger(),
        "Skipped roughness-anchor existing-edge replacement event %d->%d role=%s "
        "because stored edge direction does not match from_id=%d to_id=%d.",
        constraint.event_a,
        constraint.event_b,
        constraint.role.c_str(),
        from_scan->GetStateId(),
        to_scan->GetStateId());
      return false;
    }
    auto * old_link = dynamic_cast<karto::LinkInfo *>(edge->GetLabel());
    if (!old_link) {
      constraint.injected = true;
      RCLCPP_WARN(
        get_logger(),
        "Skipped roughness-anchor existing-edge replacement event %d->%d role=%s "
        "because the existing edge label is not karto::LinkInfo.",
        constraint.event_a,
        constraint.event_b,
        constraint.role.c_str());
      return false;
    }
    old_edge_measurement = old_link->GetPoseDifference();
    old_edge_covariance = old_link->GetCovariance();
    solver->RemoveConstraint(
      edge->GetSource()->GetObject()->GetUniqueId(),
      edge->GetTarget()->GetObject()->GetUniqueId());
    delete old_link;
    replaced_existing_edge = true;
  }

  edge->SetLabel(new karto::LinkInfo(from_corrected, to_expected, edge_covariance));
  solver->AddConstraint(edge);
  smapper_->getMapper()->CorrectPoses();
  injected_edges_.insert(edge_key);
  constraint.injected = true;
  const karto::Pose2 edge_measurement = relativePose(from_corrected, to_expected);

  if (replaced_existing_edge) {
    RCLCPP_INFO(
      get_logger(),
      "Replaced roughness-anchor existing graph edge event %d->%d role=%s "
      "source_id=%d target_id=%d from_id=%d to_id=%d graph_odom_gap=%.3f "
      "measurement_mode=%s old_dx=%.4f old_dy=%.4f old_yaw_deg=%.3f "
      "new_dx=%.4f new_dy=%.4f new_yaw_deg=%.3f "
      "old_covariance_diag=%.6f/%.6f/%.6f new_covariance_diag=%.6f/%.6f/%.6f.",
      constraint.event_a,
      constraint.event_b,
      constraint.role.c_str(),
      source_id,
      target_id,
      from_scan->GetStateId(),
      to_scan->GetStateId(),
      graph_odom_gap_m,
      measurement_mode_.c_str(),
      old_edge_measurement.GetX(),
      old_edge_measurement.GetY(),
      old_edge_measurement.GetHeading() * 180.0 / M_PI,
      edge_measurement.GetX(),
      edge_measurement.GetY(),
      edge_measurement.GetHeading() * 180.0 / M_PI,
      old_edge_covariance(0, 0),
      old_edge_covariance(1, 1),
      old_edge_covariance(2, 2),
      edge_covariance(0, 0),
      edge_covariance(1, 1),
      edge_covariance(2, 2));
    return true;
  }

  RCLCPP_INFO(
    get_logger(),
    "Injected roughness-anchor graph edge event %d->%d role=%s source_id=%d target_id=%d "
    "from_id=%d to_id=%d source_dt=%.3f target_dt=%.3f iou=%.3f nn_p50=%.3f "
    "source_odom_gap=%.3f target_odom_gap=%.3f graph_odom_gap=%.3f input_odom_gap=%.3f "
    "measurement_mode=%s measurement_dx=%.4f measurement_dy=%.4f "
    "measurement_yaw_deg=%.3f covariance_diag=%.6f/%.6f/%.6f.",
    constraint.event_a,
    constraint.event_b,
    constraint.role.c_str(),
    source_id,
    target_id,
    from_scan->GetStateId(),
    to_scan->GetStateId(),
    source_match.dt_s,
    target_match.dt_s,
    constraint.scan_iou,
    constraint.scan_nn_p50_m,
    source_match.odom_gap_m,
    target_match.odom_gap_m,
    graph_odom_gap_m,
    constraint.input_odom_gap_m,
    measurement_mode_.c_str(),
    edge_measurement.GetX(),
    edge_measurement.GetY(),
    edge_measurement.GetHeading() * 180.0 / M_PI,
    edge_covariance(0, 0),
    edge_covariance(1, 1),
    edge_covariance(2, 2));
  return true;
}

RoughnessAnchorSlamToolbox::VertexMatch RoughnessAnchorSlamToolbox::findNearestVertex(
  double stamp_s,
  double odom_x_m,
  double odom_y_m) const
{
  VertexMatch best;
  if (!smapper_ || !smapper_->getMapper() || !smapper_->getMapper()->GetGraph()) {
    return best;
  }

  const bool has_odom_target = std::isfinite(odom_x_m) && std::isfinite(odom_y_m);
  const auto & vertices_by_sensor = smapper_->getMapper()->GetGraph()->GetVertices();
  for (const auto & sensor_vertices : vertices_by_sensor) {
    for (const auto & id_vertex : sensor_vertices.second) {
      auto * vertex = id_vertex.second;
      if (!vertex || !vertex->GetObject()) {
        continue;
      }
      const double dt_s = std::abs(vertex->GetObject()->GetTime() - stamp_s);
      double odom_gap_m = std::numeric_limits<double>::infinity();
      if (has_odom_target) {
        const karto::Pose2 odom_pose = vertex->GetObject()->GetOdometricPose();
        odom_gap_m = std::hypot(odom_pose.GetX() - odom_x_m, odom_pose.GetY() - odom_y_m);
        if (odom_gap_m > max_vertex_odom_gap_m_) {
          continue;
        }
      }
      // A repeated route can contain several vertices at nearly identical odom
      // coordinates. Anchor timestamps identify the intended traversal; odom is
      // a consistency gate and only breaks timestamp ties.
      const bool earlier_in_time = dt_s + 1.0e-6 < best.dt_s;
      const bool same_time_better_odom =
        std::abs(dt_s - best.dt_s) <= 1.0e-6 && odom_gap_m < best.odom_gap_m;
      if (earlier_in_time || same_time_better_odom) {
        best.vertex = vertex;
        best.dt_s = dt_s;
        best.odom_gap_m = odom_gap_m;
      }
    }
  }
  return best;
}

RoughnessAnchorSlamToolbox::BaseScanSet RoughnessAnchorSlamToolbox::collectBaseScans(
  karto::LocalizedRangeScan * anchor_scan,
  karto::LocalizedRangeScan * graph_anchor_scan,
  karto::LocalizedRangeScan * moving_graph_scan) const
{
  BaseScanSet base;
  if (!anchor_scan || !smapper_ || !smapper_->getMapper() ||
    !smapper_->getMapper()->GetGraph())
  {
    return base;
  }

  base.scans.push_back(anchor_scan);
  if (scan_match_base_half_window_ <= 0) {
    return base;
  }

  if (!graph_anchor_scan || graph_anchor_scan->GetStateId() < 0) {
    return base;
  }

  std::vector<std::pair<int, karto::LocalizedRangeScan *>> ordered;
  const auto & vertices_by_sensor = smapper_->getMapper()->GetGraph()->GetVertices();
  for (const auto & sensor_vertices : vertices_by_sensor) {
    for (const auto & id_vertex : sensor_vertices.second) {
      auto * vertex = id_vertex.second;
      auto * scan = vertex ? vertex->GetObject() : nullptr;
      if (!scan || scan->GetSensorName() != anchor_scan->GetSensorName()) {
        continue;
      }
      if (
        scan == graph_anchor_scan ||
        scan->GetUniqueId() == graph_anchor_scan->GetUniqueId() ||
        scan->GetStateId() == graph_anchor_scan->GetStateId())
      {
        continue;
      }
      if (
        moving_graph_scan &&
        (scan == moving_graph_scan ||
        scan->GetUniqueId() == moving_graph_scan->GetUniqueId() ||
        scan->GetStateId() == moving_graph_scan->GetStateId()))
      {
        continue;
      }
      if (
        std::abs(scan->GetStateId() - graph_anchor_scan->GetStateId()) >
        scan_match_base_half_window_)
      {
        continue;
      }
      if (
        std::abs(scan->GetTime() - anchor_scan->GetTime()) >
        scan_match_max_base_dt_s_)
      {
        continue;
      }
      ordered.emplace_back(scan->GetStateId(), scan);
    }
  }
  std::sort(
    ordered.begin(), ordered.end(),
    [](const auto & first, const auto & second) {return first.first < second.first;});
  for (const auto & state_scan : ordered) {
    auto * graph_scan = state_scan.second;
    auto local_scan = std::make_unique<karto::LocalizedRangeScan>(
      graph_scan->GetSensorName(), graph_scan->GetRangeReadingsVector());
    local_scan->SetUniqueId(graph_scan->GetUniqueId());
    local_scan->SetStateId(graph_scan->GetStateId());
    local_scan->SetTime(graph_scan->GetTime());
    local_scan->SetOdometricPose(graph_scan->GetOdometricPose());
    const karto::Pose2 anchor_to_neighbor = relativePose(
      graph_anchor_scan->GetOdometricPose(), graph_scan->GetOdometricPose());
    local_scan->SetCorrectedPose(
      composePose(anchor_scan->GetCorrectedPose(), anchor_to_neighbor));
    base.scans.push_back(local_scan.get());
    base.owned_scans.push_back(std::move(local_scan));
  }
  return base;
}

RoughnessAnchorSlamToolbox::BaseScanSet
RoughnessAnchorSlamToolbox::makeCachedRawBaseScans(
  const std::vector<CachedRoughnessAnchorScan> & raw_scans,
  karto::LocalizedRangeScan * graph_anchor_scan) const
{
  BaseScanSet base;
  if (!graph_anchor_scan) {
    return base;
  }

  for (const auto & cached : raw_scans) {
    auto local_scan = makeCachedAnchorScan(
      cached.scan, graph_anchor_scan, cached.odom_pose);
    if (!local_scan) {
      continue;
    }
    base.scans.push_back(local_scan.get());
    base.owned_scans.push_back(std::move(local_scan));
  }
  return base;
}

RoughnessAnchorSlamToolbox::ScanMatchResult RoughnessAnchorSlamToolbox::matchScanToAnchor(
  karto::LocalizedRangeScan * moving_scan,
  karto::LocalizedRangeScan * anchor_scan,
  karto::LocalizedRangeScan * graph_anchor_scan,
  karto::LocalizedRangeScan * moving_graph_scan) const
{
  const BaseScanSet base = collectBaseScans(
    anchor_scan, graph_anchor_scan, moving_graph_scan);
  return matchScanToBase(moving_scan, base.scans);
}

RoughnessAnchorSlamToolbox::ScanMatchResult RoughnessAnchorSlamToolbox::matchScanToBase(
  karto::LocalizedRangeScan * moving_scan,
  const karto::LocalizedRangeScanVector & base_scans) const
{
  ScanMatchResult result;
  if (!moving_scan || !smapper_ || !smapper_->getMapper()) {
    result.rejection_reason = "missing_scan_or_mapper";
    return result;
  }

  auto * mapper = smapper_->getMapper();
  auto * coarse_matcher = mapper->GetLoopScanMatcher();
  auto * fine_matcher = mapper->GetSequentialScanMatcher();
  result.base_scan_count = base_scans.size();
  if (!coarse_matcher || !fine_matcher || base_scans.empty()) {
    result.rejection_reason = "missing_matcher_or_base_scan";
    return result;
  }

  karto::LocalizedRangeScan temporary_scan(
    moving_scan->GetSensorName(), moving_scan->GetRangeReadingsVector());
  temporary_scan.SetUniqueId(moving_scan->GetUniqueId());
  temporary_scan.SetTime(moving_scan->GetTime());
  temporary_scan.SetStateId(moving_scan->GetStateId());
  temporary_scan.SetCorrectedPose(moving_scan->GetCorrectedPose());

  karto::Pose2 matched_sensor_pose;
  karto::Matrix3 coarse_covariance;
  result.coarse_response = coarse_matcher->MatchScan(
    &temporary_scan,
    base_scans,
    matched_sensor_pose,
    coarse_covariance,
    false,
    false);
  result.coarse_variance_x = coarse_covariance(0, 0);
  result.coarse_variance_y = coarse_covariance(1, 1);
  if (
    !std::isfinite(result.coarse_response) ||
    !std::isfinite(result.coarse_variance_x) ||
    !std::isfinite(result.coarse_variance_y))
  {
    result.rejection_reason = "non_finite_coarse_result";
    return result;
  }
  if (result.coarse_response <= scan_match_min_coarse_response_) {
    result.rejection_reason = "coarse_response";
    return result;
  }
  if (
    result.coarse_variance_x >= scan_match_max_coarse_variance_ ||
    result.coarse_variance_y >= scan_match_max_coarse_variance_)
  {
    result.rejection_reason = "coarse_variance";
    return result;
  }

  temporary_scan.SetSensorPose(matched_sensor_pose);
  karto::Matrix3 fine_covariance;
  result.fine_response = fine_matcher->MatchScan(
    &temporary_scan,
    base_scans,
    matched_sensor_pose,
    fine_covariance,
    false,
    true);
  if (!std::isfinite(result.fine_response)) {
    result.rejection_reason = "non_finite_fine_response";
    return result;
  }
  if (result.fine_response < scan_match_min_fine_response_) {
    result.rejection_reason = "fine_response";
    return result;
  }

  result.matched_robot_pose = moving_scan->GetCorrectedAt(matched_sensor_pose);
  result.translation_innovation_m = poseTranslationDistance(
    result.matched_robot_pose, moving_scan->GetCorrectedPose());
  result.yaw_innovation_rad = std::abs(
    normalizedAngle(
      result.matched_robot_pose.GetHeading() -
      moving_scan->GetCorrectedPose().GetHeading()));
  if (
    !std::isfinite(result.matched_robot_pose.GetX()) ||
    !std::isfinite(result.matched_robot_pose.GetY()) ||
    !std::isfinite(result.matched_robot_pose.GetHeading()))
  {
    result.rejection_reason = "non_finite_matched_pose";
    return result;
  }
  if (result.translation_innovation_m > scan_match_max_translation_innovation_m_) {
    result.rejection_reason = "translation_innovation";
    return result;
  }
  if (result.yaw_innovation_rad > scan_match_max_yaw_innovation_rad_) {
    result.rejection_reason = "yaw_innovation";
    return result;
  }

  result.covariance = fine_covariance;
  if (!regularizeCovariance(result.covariance)) {
    result.rejection_reason = "invalid_covariance";
    return result;
  }
  result.accepted = true;
  return result;
}

bool RoughnessAnchorSlamToolbox::makeScanMatchMeasurement(
  const RoughnessAnchorConstraint & constraint,
  karto::LocalizedRangeScan * from_scan,
  karto::LocalizedRangeScan * to_scan,
  karto::Pose2 & matched_to_pose,
  karto::Matrix3 & covariance,
  ScanMatchResult & forward,
  ScanMatchResult & reverse,
  std::string & rejection_reason) const
{
  const bool from_is_target =
    std::abs(from_scan->GetTime() - constraint.target_stamp_s) <=
    std::abs(from_scan->GetTime() - constraint.source_stamp_s);
  const auto & from_raw_message =
    from_is_target ? constraint.target_raw_scan : constraint.source_raw_scan;
  const auto & to_raw_message =
    from_is_target ? constraint.source_raw_scan : constraint.target_raw_scan;
  const auto & from_raw_window =
    from_is_target ? constraint.target_raw_window : constraint.source_raw_window;
  const auto & to_raw_window =
    from_is_target ? constraint.source_raw_window : constraint.target_raw_window;
  const karto::Pose2 from_anchor_odom = from_is_target ?
    karto::Pose2(
    constraint.target_odom_x_m,
    constraint.target_odom_y_m,
    constraint.target_odom_yaw_rad) :
    karto::Pose2(
    constraint.source_odom_x_m,
    constraint.source_odom_y_m,
    constraint.source_odom_yaw_rad);
  const karto::Pose2 to_anchor_odom = from_is_target ?
    karto::Pose2(
    constraint.source_odom_x_m,
    constraint.source_odom_y_m,
    constraint.source_odom_yaw_rad) :
    karto::Pose2(
    constraint.target_odom_x_m,
    constraint.target_odom_y_m,
    constraint.target_odom_yaw_rad);

  std::unique_ptr<karto::LocalizedRangeScan> cached_from_scan;
  std::unique_ptr<karto::LocalizedRangeScan> cached_to_scan;
  karto::LocalizedRangeScan * match_from_scan = from_scan;
  karto::LocalizedRangeScan * match_to_scan = to_scan;
  bool using_cached_anchor_scans = false;
  if (scan_match_use_cached_anchor_scans_) {
    const bool have_raw_messages = from_raw_message && to_raw_message;
    const bool have_raw_odom =
      std::isfinite(from_anchor_odom.GetX()) &&
      std::isfinite(from_anchor_odom.GetY()) &&
      std::isfinite(from_anchor_odom.GetHeading()) &&
      std::isfinite(to_anchor_odom.GetX()) &&
      std::isfinite(to_anchor_odom.GetY()) &&
      std::isfinite(to_anchor_odom.GetHeading());
    if (have_raw_messages && have_raw_odom) {
      cached_from_scan = makeCachedAnchorScan(
        from_raw_message, from_scan, from_anchor_odom);
      cached_to_scan = makeCachedAnchorScan(
        to_raw_message, to_scan, to_anchor_odom);
      if (cached_from_scan && cached_to_scan) {
        match_from_scan = cached_from_scan.get();
        match_to_scan = cached_to_scan.get();
        using_cached_anchor_scans = true;
      }
    }
    if (scan_match_require_cached_anchor_scans_ && !using_cached_anchor_scans) {
      rejection_reason = "missing_cached_anchor_scan";
      return false;
    }
  }

  BaseScanSet forward_raw_base;
  BaseScanSet reverse_raw_base;
  if (scan_match_raw_window_half_duration_s_ > 0.0) {
    if (
      static_cast<int>(from_raw_window.size()) <
      scan_match_raw_window_min_scans_ ||
      static_cast<int>(to_raw_window.size()) <
      scan_match_raw_window_min_scans_)
    {
      rejection_reason = "insufficient_cached_raw_window";
      return false;
    }
    forward_raw_base = makeCachedRawBaseScans(from_raw_window, from_scan);
    reverse_raw_base = makeCachedRawBaseScans(to_raw_window, to_scan);
    if (
      static_cast<int>(forward_raw_base.scans.size()) <
      scan_match_raw_window_min_scans_ ||
      static_cast<int>(reverse_raw_base.scans.size()) <
      scan_match_raw_window_min_scans_)
    {
      rejection_reason = "failed_cached_raw_window_conversion";
      return false;
    }
    forward = matchScanToBase(match_to_scan, forward_raw_base.scans);
  } else {
    forward = matchScanToAnchor(
      match_to_scan, match_from_scan, from_scan, to_scan);
  }
  if (!forward.accepted) {
    rejection_reason = "forward_" + forward.rejection_reason;
    return false;
  }

  if (scan_match_require_bidirectional_) {
    if (scan_match_raw_window_half_duration_s_ > 0.0) {
      reverse = matchScanToBase(match_from_scan, reverse_raw_base.scans);
    } else {
      reverse = matchScanToAnchor(
        match_from_scan, match_to_scan, to_scan, from_scan);
    }
    if (!reverse.accepted) {
      rejection_reason = "reverse_" + reverse.rejection_reason;
      return false;
    }
    const karto::Pose2 forward_relative = relativePose(
      match_from_scan->GetCorrectedPose(), forward.matched_robot_pose);
    const karto::Pose2 reverse_relative = relativePose(
      reverse.matched_robot_pose, match_to_scan->GetCorrectedPose());
    const double translation_error_m = poseTranslationDistance(
      forward_relative, reverse_relative);
    const double yaw_error_rad = std::abs(
      normalizedAngle(
        forward_relative.GetHeading() - reverse_relative.GetHeading()));
    forward.bidirectional_translation_error_m = translation_error_m;
    forward.bidirectional_yaw_error_rad = yaw_error_rad;
    if (translation_error_m > scan_match_max_bidirectional_translation_error_m_) {
      rejection_reason = "bidirectional_translation";
      return false;
    }
    if (yaw_error_rad > scan_match_max_bidirectional_yaw_error_rad_) {
      rejection_reason = "bidirectional_yaw";
      return false;
    }
  }

  if (using_cached_anchor_scans) {
    const karto::Pose2 to_graph_to_anchor = relativePose(
      to_scan->GetOdometricPose(), to_anchor_odom);
    matched_to_pose = composePose(
      forward.matched_robot_pose, inversePose(to_graph_to_anchor));
  } else {
    matched_to_pose = forward.matched_robot_pose;
  }
  covariance = forward.covariance;
  return true;
}

std::unique_ptr<karto::LocalizedRangeScan>
RoughnessAnchorSlamToolbox::makeCachedAnchorScan(
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & raw_scan,
  karto::LocalizedRangeScan * graph_scan,
  const karto::Pose2 & anchor_odom_pose) const
{
  if (!raw_scan || !graph_scan) {
    return nullptr;
  }
  const auto metadata = lasers_.find(raw_scan->header.frame_id);
  if (metadata == lasers_.end()) {
    return nullptr;
  }

  const std::vector<double> readings = laser_utils::scanToReadings(
    *raw_scan, metadata->second.isInverted());
  auto anchor_scan = std::make_unique<karto::LocalizedRangeScan>(
    graph_scan->GetSensorName(), readings);
  const karto::Pose2 graph_to_anchor = relativePose(
    graph_scan->GetOdometricPose(), anchor_odom_pose);
  const karto::Pose2 anchor_corrected = composePose(
    graph_scan->GetCorrectedPose(), graph_to_anchor);
  anchor_scan->SetUniqueId(graph_scan->GetUniqueId());
  anchor_scan->SetStateId(-1);
  anchor_scan->SetTime(rclcpp::Time(raw_scan->header.stamp).seconds());
  anchor_scan->SetOdometricPose(anchor_odom_pose);
  anchor_scan->SetCorrectedPose(anchor_corrected);
  return anchor_scan;
}

bool RoughnessAnchorSlamToolbox::regularizeCovariance(karto::Matrix3 & covariance) const
{
  Eigen::Matrix3d matrix;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const double value = covariance(row, column);
      if (!std::isfinite(value)) {
        return false;
      }
      matrix(row, column) = value;
    }
  }
  matrix = 0.5 * (matrix + matrix.transpose()).eval();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(matrix);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return false;
  }
  Eigen::Vector3d eigenvalues = solver.eigenvalues() * scan_match_covariance_inflation_;
  const double minimum_variance = std::min(
    scan_match_min_translation_sigma_m_ * scan_match_min_translation_sigma_m_,
    scan_match_min_yaw_sigma_rad_ * scan_match_min_yaw_sigma_rad_);
  eigenvalues = eigenvalues.cwiseMax(minimum_variance);
  matrix = solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
  matrix(0, 0) = std::max(
    matrix(0, 0),
    scan_match_min_translation_sigma_m_ * scan_match_min_translation_sigma_m_);
  matrix(1, 1) = std::max(
    matrix(1, 1),
    scan_match_min_translation_sigma_m_ * scan_match_min_translation_sigma_m_);
  matrix(2, 2) = std::max(
    matrix(2, 2),
    scan_match_min_yaw_sigma_rad_ * scan_match_min_yaw_sigma_rad_);
  matrix = 0.5 * (matrix + matrix.transpose()).eval();

  Eigen::LLT<Eigen::Matrix3d> decomposition(matrix);
  if (decomposition.info() != Eigen::Success || !matrix.allFinite()) {
    return false;
  }
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      covariance(row, column) = matrix(row, column);
    }
  }
  return true;
}

std::size_t RoughnessAnchorSlamToolbox::countGraphVertices() const
{
  std::size_t count = 0;
  if (!smapper_ || !smapper_->getMapper() || !smapper_->getMapper()->GetGraph()) {
    return count;
  }

  const auto & vertices_by_sensor = smapper_->getMapper()->GetGraph()->GetVertices();
  for (const auto & sensor_vertices : vertices_by_sensor) {
    count += sensor_vertices.second.size();
  }
  return count;
}

void RoughnessAnchorSlamToolbox::logPendingConstraintStatus() const
{
  for (const auto & constraint : constraints_) {
    if (constraint.injected) {
      continue;
    }
    const VertexMatch source_match = findNearestVertex(
      constraint.source_stamp_s,
      constraint.source_odom_x_m,
      constraint.source_odom_y_m);
    const VertexMatch target_match = findNearestVertex(
      constraint.target_stamp_s,
      constraint.target_odom_x_m,
      constraint.target_odom_y_m);
    auto * source_scan =
      source_match.vertex ? source_match.vertex->GetObject() : nullptr;
    auto * target_scan =
      target_match.vertex ? target_match.vertex->GetObject() : nullptr;
    const int source_id = source_scan ? source_scan->GetStateId() : -1;
    const int target_id = target_scan ? target_scan->GetStateId() : -1;
    const int state_id_separation =
      source_scan && target_scan ? std::abs(source_id - target_id) : -1;
    const double graph_odom_gap_m =
      source_scan && target_scan ?
      poseTranslationDistance(
      source_scan->GetOdometricPose(), target_scan->GetOdometricPose()) :
      std::numeric_limits<double>::infinity();
    RCLCPP_INFO(
      get_logger(),
      "Roughness-anchor pending event %d->%d role=%s graph_vertices=%zu "
      "source_stamp=%.3f source_dt=%.3f target_stamp=%.3f target_dt=%.3f "
      "source_odom_gap=%.3f target_odom_gap=%.3f max_dt=%.3f max_vertex_odom_gap=%.3f "
      "source_id=%d target_id=%d state_id_separation=%d graph_odom_gap=%.3f "
      "target_raw=%s target_raw_dt=%.6f source_raw=%s source_raw_dt=%.6f "
      "iou=%.3f nn_p50=%.3f.",
      constraint.event_a,
      constraint.event_b,
      constraint.role.c_str(),
      countGraphVertices(),
      constraint.source_stamp_s,
      source_match.dt_s,
      constraint.target_stamp_s,
      target_match.dt_s,
      source_match.odom_gap_m,
      target_match.odom_gap_m,
      max_stamp_dt_s_,
      max_vertex_odom_gap_m_,
      source_id,
      target_id,
      state_id_separation,
      graph_odom_gap_m,
      constraint.target_raw_scan ? "yes" : "no",
      constraint.target_raw_scan_dt_s,
      constraint.source_raw_scan ? "yes" : "no",
      constraint.source_raw_scan_dt_s,
      constraint.scan_iou,
      constraint.scan_nn_p50_m);
    return;
  }
}

karto::Matrix3 RoughnessAnchorSlamToolbox::makeAnchorCovariance() const
{
  karto::Matrix3 covariance;
  covariance.Clear();
  covariance(0, 0) = translation_sigma_m_ * translation_sigma_m_;
  covariance(1, 1) = translation_sigma_m_ * translation_sigma_m_;
  covariance(2, 2) = yaw_sigma_rad_ * yaw_sigma_rad_;
  return covariance;
}

}  // namespace slam_bringup
