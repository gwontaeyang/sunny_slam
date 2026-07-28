#ifndef SLAM_BRINGUP__ROUGHNESS_ANCHOR_SLAM_TOOLBOX_HPP_
#define SLAM_BRINGUP__ROUGHNESS_ANCHOR_SLAM_TOOLBOX_HPP_

#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "sensor_msgs/msg/laser_scan.hpp"
#include "slam_toolbox/slam_toolbox_sync.hpp"

namespace karto
{

// ROS Humble's karto shared library does not export this template specialization.
template<>
kt_double ScanMatcher::MatchScan<LocalizedRangeScanVector>(
  LocalizedRangeScan * pScan,
  const LocalizedRangeScanVector & rBaseScans,
  Pose2 & rMean,
  Matrix3 & rCovariance,
  kt_bool doPenalize,
  kt_bool doRefineMatch);

}  // namespace karto

namespace slam_bringup
{

struct CachedRoughnessAnchorScan
{
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
  karto::Pose2 odom_pose;
  double stamp_s{0.0};
};

struct RoughnessAnchorConstraint
{
  int event_a{-1};
  int event_b{-1};
  std::string role;
  double target_stamp_s{0.0};
  double source_stamp_s{0.0};
  double scan_iou{0.0};
  double scan_nn_p50_m{0.0};
  double scan_yaw_deg{0.0};
  double input_odom_gap_m{std::numeric_limits<double>::quiet_NaN()};
  double input_residual_m{std::numeric_limits<double>::quiet_NaN()};
  double pair_residual_m{std::numeric_limits<double>::quiet_NaN()};
  double target_odom_x_m{std::numeric_limits<double>::quiet_NaN()};
  double target_odom_y_m{std::numeric_limits<double>::quiet_NaN()};
  double target_odom_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double source_odom_x_m{std::numeric_limits<double>::quiet_NaN()};
  double source_odom_y_m{std::numeric_limits<double>::quiet_NaN()};
  double source_odom_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  sensor_msgs::msg::LaserScan::ConstSharedPtr target_raw_scan;
  sensor_msgs::msg::LaserScan::ConstSharedPtr source_raw_scan;
  std::vector<CachedRoughnessAnchorScan> target_raw_window;
  std::vector<CachedRoughnessAnchorScan> source_raw_window;
  double target_raw_scan_dt_s{std::numeric_limits<double>::infinity()};
  double source_raw_scan_dt_s{std::numeric_limits<double>::infinity()};
  std::pair<int, int> last_graph_odom_reject_ids{-1, -1};
  std::pair<int, int> last_scan_match_reject_ids{-1, -1};
  bool injected{false};
};

class RoughnessAnchorSlamToolbox : public slam_toolbox::SynchronousSlamToolbox
{
public:
  explicit RoughnessAnchorSlamToolbox(rclcpp::NodeOptions options);

  void configure() override;

protected:
  void laserCallback(sensor_msgs::msg::LaserScan::ConstSharedPtr scan) override;

private:
  struct VertexMatch
  {
    karto::Vertex<karto::LocalizedRangeScan> * vertex{nullptr};
    double dt_s{std::numeric_limits<double>::infinity()};
    double odom_gap_m{std::numeric_limits<double>::infinity()};
  };

  struct ScanMatchResult
  {
    bool accepted{false};
    karto::Pose2 matched_robot_pose;
    karto::Matrix3 covariance;
    double coarse_response{0.0};
    double fine_response{0.0};
    double coarse_variance_x{std::numeric_limits<double>::infinity()};
    double coarse_variance_y{std::numeric_limits<double>::infinity()};
    double translation_innovation_m{std::numeric_limits<double>::infinity()};
    double yaw_innovation_rad{std::numeric_limits<double>::infinity()};
    double bidirectional_translation_error_m{std::numeric_limits<double>::infinity()};
    double bidirectional_yaw_error_rad{std::numeric_limits<double>::infinity()};
    std::size_t base_scan_count{0};
    std::string rejection_reason;
  };

  struct BaseScanSet
  {
    std::vector<std::unique_ptr<karto::LocalizedRangeScan>> owned_scans;
    karto::LocalizedRangeScanVector scans;
  };

  void loadRoughnessAnchorParameters();
  void loadConstraintsFromCsv(const std::string & csv_path);
  void cacheConstraintScan(sensor_msgs::msg::LaserScan::ConstSharedPtr scan);
  void tryInjectPendingConstraints();
  bool tryInjectConstraint(RoughnessAnchorConstraint & constraint);
  VertexMatch findNearestVertex(double stamp_s, double odom_x_m, double odom_y_m) const;
  BaseScanSet collectBaseScans(
    karto::LocalizedRangeScan * anchor_scan,
    karto::LocalizedRangeScan * graph_anchor_scan,
    karto::LocalizedRangeScan * moving_graph_scan) const;
  BaseScanSet makeCachedRawBaseScans(
    const std::vector<CachedRoughnessAnchorScan> & raw_scans,
    karto::LocalizedRangeScan * graph_anchor_scan) const;
  ScanMatchResult matchScanToBase(
    karto::LocalizedRangeScan * moving_scan,
    const karto::LocalizedRangeScanVector & base_scans) const;
  ScanMatchResult matchScanToAnchor(
    karto::LocalizedRangeScan * moving_scan,
    karto::LocalizedRangeScan * anchor_scan,
    karto::LocalizedRangeScan * graph_anchor_scan,
    karto::LocalizedRangeScan * moving_graph_scan) const;
  bool makeScanMatchMeasurement(
    const RoughnessAnchorConstraint & constraint,
    karto::LocalizedRangeScan * from_scan,
    karto::LocalizedRangeScan * to_scan,
    karto::Pose2 & matched_to_pose,
    karto::Matrix3 & covariance,
    ScanMatchResult & forward,
    ScanMatchResult & reverse,
    std::string & rejection_reason) const;
  std::unique_ptr<karto::LocalizedRangeScan> makeCachedAnchorScan(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & raw_scan,
    karto::LocalizedRangeScan * graph_scan,
    const karto::Pose2 & anchor_odom_pose) const;
  bool regularizeCovariance(karto::Matrix3 & covariance) const;
  std::size_t countGraphVertices() const;
  void logPendingConstraintStatus() const;
  karto::Matrix3 makeAnchorCovariance() const;

  bool enable_constraints_{false};
  bool inject_constraints_{true};
  bool replace_existing_edge_{false};
  std::string constraints_csv_;
  std::string measurement_mode_{"scan_match"};
  int filter_event_a_{-1};
  int filter_event_b_{-1};
  double max_stamp_dt_s_{8.0};
  double min_scan_iou_{0.15};
  double max_scan_nn_p50_m_{0.10};
  double max_abs_scan_yaw_deg_{5.0};
  double max_input_odom_gap_m_{0.10};
  double min_pair_residual_m_{0.05};
  double max_vertex_odom_gap_m_{0.50};
  double max_graph_odom_gap_m_{0.05};
  double scan_match_max_graph_odom_gap_m_{0.75};
  bool scan_match_use_cached_anchor_scans_{true};
  bool scan_match_require_cached_anchor_scans_{true};
  double scan_match_raw_scan_max_dt_s_{0.10};
  double scan_match_raw_window_half_duration_s_{0.0};
  int scan_match_raw_window_min_scans_{3};
  int scan_match_raw_window_max_scans_{9};
  double scan_match_raw_window_min_separation_s_{0.04};
  bool scan_match_raw_window_stable_side_only_{false};
  int scan_match_base_half_window_{0};
  double scan_match_max_base_dt_s_{8.0};
  double scan_match_min_coarse_response_{0.70};
  double scan_match_min_fine_response_{0.55};
  double scan_match_max_coarse_variance_{0.16};
  double scan_match_max_translation_innovation_m_{0.75};
  double scan_match_max_yaw_innovation_rad_{0.17453292519943295};
  bool scan_match_require_bidirectional_{true};
  double scan_match_max_bidirectional_translation_error_m_{0.15};
  double scan_match_max_bidirectional_yaw_error_rad_{0.05235987755982989};
  double scan_match_covariance_inflation_{2.0};
  double scan_match_min_translation_sigma_m_{0.05};
  double scan_match_min_yaw_sigma_rad_{0.03490658503988659};
  double translation_sigma_m_{0.10};
  double yaw_sigma_rad_{1.0};
  int min_state_id_separation_{5};
  int max_constraint_pairs_{8};
  bool require_complete_anchor_pair_{true};
  bool update_map_after_constraints_{true};
  bool constraints_loaded_{false};
  bool load_error_reported_{false};
  std::size_t anchor_timer_tick_{0};
  double latest_raw_scan_stamp_s_{-std::numeric_limits<double>::infinity()};
  rclcpp::TimerBase::SharedPtr anchor_timer_;
  std::vector<RoughnessAnchorConstraint> constraints_;
  std::set<std::pair<int, int>> injected_edges_;
};

}  // namespace slam_bringup

#endif  // SLAM_BRINGUP__ROUGHNESS_ANCHOR_SLAM_TOOLBOX_HPP_
