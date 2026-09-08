#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "karto_sdk/Mapper.h"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "slam_2d_core/pose_graph_constraint_replay.hpp"

namespace
{

using Constraint = slam_2d_core::PoseGraphReplayConstraint;

struct Residual
{
  double x_m{std::numeric_limits<double>::quiet_NaN()};
  double y_m{std::numeric_limits<double>::quiet_NaN()};
  double yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double translation_m{std::numeric_limits<double>::quiet_NaN()};
  double covariance_weighted_sq{std::numeric_limits<double>::quiet_NaN()};
};

struct ConstraintRuntime
{
  Constraint input;
  karto::LocalizedRangeScan * source{nullptr};
  karto::LocalizedRangeScan * target{nullptr};
  Residual before;
  Residual after;
};

struct RclShutdownGuard
{
  ~RclShutdownGuard()
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

void printUsage(const char * executable)
{
  std::cerr
    << "Usage: " << executable
    << " <pose-graph-stem> <constraint-csv> <output-directory>"
    << " [--diagnostics-only | map-resolution-m min-pass-through occupancy-threshold]\n";
}

double normalizedAngle(const double angle_rad)
{
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

karto::Pose2 relativePose(const karto::Pose2 & source, const karto::Pose2 & target)
{
  karto::Transform transform(source, karto::Pose2());
  return transform.TransformPose(target);
}

karto::Pose2 composePose(const karto::Pose2 & source, const karto::Pose2 & relative)
{
  karto::Transform transform(source);
  return transform.TransformPose(relative);
}

karto::Matrix3 covarianceMatrix(const std::array<double, 6> & values)
{
  karto::Matrix3 covariance;
  covariance(0, 0) = values[0];
  covariance(0, 1) = values[1];
  covariance(1, 0) = values[1];
  covariance(0, 2) = values[2];
  covariance(2, 0) = values[2];
  covariance(1, 1) = values[3];
  covariance(1, 2) = values[4];
  covariance(2, 1) = values[4];
  covariance(2, 2) = values[5];
  return covariance;
}

karto::Matrix3 covarianceInWorldFrame(
  const karto::Matrix3 & source_frame_covariance,
  const double source_yaw_rad)
{
  karto::Matrix3 rotation;
  rotation.FromAxisAngle(0.0, 0.0, 1.0, source_yaw_rad);
  return rotation * source_frame_covariance * rotation.Transpose();
}

Residual constraintResidual(
  const karto::LocalizedRangeScan * source,
  const karto::LocalizedRangeScan * target,
  const Constraint & constraint)
{
  Residual residual;
  if (source == nullptr || target == nullptr) {
    return residual;
  }
  const karto::Pose2 expected = relativePose(
    source->GetCorrectedPose(), target->GetCorrectedPose());
  residual.x_m = expected.GetX() - constraint.relative_x_m;
  residual.y_m = expected.GetY() - constraint.relative_y_m;
  residual.yaw_rad = normalizedAngle(expected.GetHeading() - constraint.relative_yaw_rad);
  residual.translation_m = std::hypot(residual.x_m, residual.y_m);

  karto::Matrix3 covariance = covarianceMatrix(constraint.covariance);
  karto::Matrix3 information;
  if (!covariance.InverseFast(information, 1.0e-14)) {
    return residual;
  }
  const double values[3] = {residual.x_m, residual.y_m, residual.yaw_rad};
  double weighted = 0.0;
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      weighted += values[row] * information(row, column) * values[column];
    }
  }
  residual.covariance_weighted_sq = weighted;
  return residual;
}

bool graphContainsPair(
  const karto::MapperGraph & graph,
  const karto::LocalizedRangeScan * source,
  const karto::LocalizedRangeScan * target)
{
  for (const auto * edge : graph.GetEdges()) {
    if (
      edge == nullptr || edge->GetSource() == nullptr || edge->GetTarget() == nullptr ||
      edge->GetSource()->GetObject() == nullptr || edge->GetTarget()->GetObject() == nullptr)
    {
      continue;
    }
    const auto * edge_source = edge->GetSource()->GetObject();
    const auto * edge_target = edge->GetTarget()->GetObject();
    if (
      (edge_source == source && edge_target == target) ||
      (edge_source == target && edge_target == source))
    {
      return true;
    }
  }
  return false;
}

bool writeTrajectory(
  const std::filesystem::path & output_path,
  const std::vector<karto::LocalizedRangeScan *> & scans,
  const std::map<std::int32_t, karto::Pose2> & before_poses)
{
  std::ofstream output(output_path);
  if (!output.is_open()) {
    return false;
  }
  output << "phase,stamp_s,state_id,unique_id,x_m,y_m,yaw_rad,odom_x_m,odom_y_m,"
         << "odom_yaw_rad\n" << std::setprecision(17);
  for (const auto * scan : scans) {
    if (scan == nullptr) {
      continue;
    }
    const auto found = before_poses.find(scan->GetStateId());
    if (found == before_poses.end()) {
      return false;
    }
    const karto::Pose2 & odom = scan->GetOdometricPose();
    const auto write_pose = [&](const char * phase, const karto::Pose2 & pose) {
        output << phase << ',' << scan->GetTime() << ',' << scan->GetStateId() << ','
               << scan->GetUniqueId() << ',' << pose.GetX() << ',' << pose.GetY() << ','
               << pose.GetHeading() << ',' << odom.GetX() << ',' << odom.GetY() << ','
               << odom.GetHeading() << '\n';
      };
    write_pose("before", found->second);
    write_pose("after", scan->GetCorrectedPose());
  }
  return output.good();
}

bool writeConstraintDiagnostics(
  const std::filesystem::path & output_path,
  const std::vector<ConstraintRuntime> & constraints)
{
  std::ofstream output(output_path);
  if (!output.is_open()) {
    return false;
  }
  output
    << "group_label,edge_label,source_state_id,target_state_id,relative_x_m,relative_y_m,"
    << "relative_yaw_rad,covariance_xx,covariance_xy,covariance_xyaw,covariance_yy,"
    << "covariance_yyaw,covariance_yawyaw,before_residual_x_m,before_residual_y_m,"
    << "before_residual_yaw_rad,before_residual_translation_m,"
    << "before_covariance_weighted_residual_sq,after_residual_x_m,after_residual_y_m,"
    << "after_residual_yaw_rad,after_residual_translation_m,"
    << "after_covariance_weighted_residual_sq\n" << std::setprecision(17);
  for (const auto & runtime : constraints) {
    const auto & input = runtime.input;
    output << input.group_label << ',' << input.edge_label << ','
           << input.source_state_id << ',' << input.target_state_id << ','
           << input.relative_x_m << ',' << input.relative_y_m << ','
           << input.relative_yaw_rad;
    for (const double value : input.covariance) {
      output << ',' << value;
    }
    const auto write_residual = [&output](const Residual & residual) {
        output << ',' << residual.x_m << ',' << residual.y_m << ',' << residual.yaw_rad
               << ',' << residual.translation_m << ',' << residual.covariance_weighted_sq;
      };
    write_residual(runtime.before);
    write_residual(runtime.after);
    output << '\n';
  }
  return output.good();
}

bool writeOccupancyMap(
  const std::filesystem::path & output_stem,
  const std::vector<karto::LocalizedRangeScan *> & scans,
  const double resolution_m,
  const std::uint32_t min_pass_through,
  const double occupancy_threshold)
{
  std::unique_ptr<karto::OccupancyGrid> grid(karto::OccupancyGrid::CreateFromScans(
      scans, resolution_m, min_pass_through, occupancy_threshold));
  if (!grid) {
    return false;
  }
  std::ofstream pgm(output_stem.string() + ".pgm", std::ios::binary);
  if (!pgm.is_open()) {
    return false;
  }
  pgm << "P5\n" << grid->GetWidth() << ' ' << grid->GetHeight() << "\n255\n";
  for (std::int32_t y = grid->GetHeight() - 1; y >= 0; --y) {
    for (std::int32_t x = 0; x < grid->GetWidth(); ++x) {
      const kt_int8u value = *grid->GetDataPointer(
        karto::Vector2<kt_int32s>(x, y));
      const unsigned char pixel = value == karto::GridStates_Occupied ? 0U :
        (value == karto::GridStates_Free ? 254U : 205U);
      pgm.write(reinterpret_cast<const char *>(&pixel), 1);
    }
  }
  if (!pgm.good()) {
    return false;
  }

  const auto & offset = grid->GetCoordinateConverter()->GetOffset();
  std::ofstream yaml(output_stem.string() + ".yaml");
  if (!yaml.is_open()) {
    return false;
  }
  yaml << std::setprecision(17)
       << "image: " << output_stem.filename().string() << ".pgm\n"
       << "mode: trinary\n"
       << "resolution: " << resolution_m << "\n"
       << "origin: [" << offset.GetX() << ", " << offset.GetY() << ", 0.0]\n"
       << "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n";
  return yaml.good();
}

bool parseMapArguments(
  const int argc,
  char ** argv,
  double & resolution_m,
  std::uint32_t & min_pass_through,
  double & occupancy_threshold,
  bool & diagnostics_only)
{
  if (argc == 4) {
    return true;
  }
  if (argc == 5 && std::string(argv[4]) == "--diagnostics-only") {
    diagnostics_only = true;
    return true;
  }
  if (argc != 7) {
    return false;
  }
  try {
    std::size_t consumed = 0U;
    resolution_m = std::stod(argv[4], &consumed);
    if (consumed != std::string(argv[4]).size()) {
      return false;
    }
    const unsigned long parsed_min = std::stoul(argv[5], &consumed);
    if (consumed != std::string(argv[5]).size() || parsed_min == 0UL) {
      return false;
    }
    min_pass_through = static_cast<std::uint32_t>(parsed_min);
    occupancy_threshold = std::stod(argv[6], &consumed);
    if (consumed != std::string(argv[6]).size()) {
      return false;
    }
  } catch (const std::exception &) {
    return false;
  }
  return std::isfinite(resolution_m) && resolution_m > 0.0 &&
         std::isfinite(occupancy_threshold) && occupancy_threshold > 0.0 &&
         occupancy_threshold < 1.0;
}

}  // namespace

int main(int argc, char ** argv)
{
  double resolution_m = 0.02;
  std::uint32_t min_pass_through = 2U;
  double occupancy_threshold = 0.1;
  bool diagnostics_only = false;
  if (
    (argc != 4 && argc != 5 && argc != 7) ||
    !parseMapArguments(
      argc, argv, resolution_m, min_pass_through, occupancy_threshold,
      diagnostics_only))
  {
    printUsage(argv[0]);
    return 2;
  }

  const std::string pose_graph_stem = argv[1];
  const std::filesystem::path constraint_path = argv[2];
  const std::filesystem::path output_directory = argv[3];
  std::ifstream constraint_input(constraint_path);
  if (!constraint_input.is_open()) {
    std::cerr << "Failed to open constraint CSV '" << constraint_path << "'\n";
    return 3;
  }
  const auto parsed = slam_2d_core::parsePoseGraphReplayConstraints(constraint_input);
  if (!parsed.errors.empty()) {
    for (const auto & error : parsed.errors) {
      std::cerr << "Constraint CSV error: " << error << '\n';
    }
    return 4;
  }

  std::cerr << "pose_graph_constraint_replay: initializing ROS and plugin loader\n";
  rclcpp::init(0, nullptr);
  RclShutdownGuard shutdown_guard;
  pluginlib::ClassLoader<karto::ScanSolver> solver_loader(
    "slam_toolbox", "karto::ScanSolver");
  std::shared_ptr<karto::ScanSolver> solver;
  const std::filesystem::path input_data = pose_graph_stem + ".data";
  if (!std::filesystem::exists(input_data)) {
    std::cerr << "Companion Karto dataset is missing: " << input_data << '\n';
    return 5;
  }
  std::cerr << "pose_graph_constraint_replay: loading sensor dataset and graph\n";
  const auto load_start = std::chrono::steady_clock::now();
  karto::Dataset dataset;
  karto::Mapper mapper;
  try {
    dataset.LoadFromFile(input_data.string());
    for (auto * object : dataset.GetLasers()) {
      auto * sensor = dynamic_cast<karto::Sensor *>(object);
      if (sensor == nullptr) {
        throw std::runtime_error("loaded Karto dataset contains a non-sensor laser object");
      }
      karto::SensorManager::GetInstance()->RegisterSensor(sensor, true);
    }
    mapper.LoadFromFile(pose_graph_stem + ".posegraph");
  } catch (const karto::Exception & exception) {
    std::cerr << "Failed to load pose graph: " << exception.GetErrorMessage() << '\n';
    return 5;
  } catch (const std::exception & exception) {
    std::cerr << "Failed to load pose graph: " << exception.what() << '\n';
    return 5;
  }

  auto scans = mapper.GetAllProcessedScans();
  const auto load_complete = std::chrono::steady_clock::now();
  std::sort(
    scans.begin(), scans.end(), [](const auto * lhs, const auto * rhs) {
      if (lhs->GetTime() == rhs->GetTime()) {
        return lhs->GetStateId() < rhs->GetStateId();
      }
      return lhs->GetTime() < rhs->GetTime();
    });
  auto * graph = mapper.GetGraph();
  if (graph == nullptr || scans.empty()) {
    std::cerr << "Loaded pose graph contains no usable graph or scans\n";
    return 6;
  }

  std::map<std::int32_t, karto::LocalizedRangeScan *> scans_by_state;
  std::map<std::int32_t, karto::Pose2> before_poses;
  for (auto * scan : scans) {
    if (scan == nullptr) {
      continue;
    }
    if (!scans_by_state.emplace(scan->GetStateId(), scan).second) {
      std::cerr << "State ID " << scan->GetStateId()
                << " is ambiguous across loaded sensors; refusing mutation\n";
      return 7;
    }
    before_poses.emplace(scan->GetStateId(), scan->GetCorrectedPose());
  }

  std::vector<ConstraintRuntime> constraints;
  constraints.reserve(parsed.constraints.size());
  for (const auto & input : parsed.constraints) {
    const auto source = scans_by_state.find(input.source_state_id);
    const auto target = scans_by_state.find(input.target_state_id);
    if (source == scans_by_state.end() || target == scans_by_state.end()) {
      std::cerr << "Constraint " << input.edge_label
                << " references a missing state ID; refusing mutation\n";
      return 8;
    }
    if (graphContainsPair(*graph, source->second, target->second)) {
      std::cerr << "Constraint " << input.edge_label
                << " duplicates an existing graph edge in either direction; refusing mutation\n";
      return 9;
    }
    ConstraintRuntime runtime;
    runtime.input = input;
    runtime.source = source->second;
    runtime.target = target->second;
    runtime.before = constraintResidual(runtime.source, runtime.target, runtime.input);
    constraints.push_back(runtime);
  }

  for (auto * edge : graph->GetEdges()) {
    if (edge == nullptr || dynamic_cast<karto::LinkInfo *>(edge->GetLabel()) == nullptr) {
      std::cerr << "Existing graph contains a null or unsupported edge; refusing mutation\n";
      return 10;
    }
  }

  std::cerr << "pose_graph_constraint_replay: configuring solver\n";
  const auto solver_start = std::chrono::steady_clock::now();
  auto solver_graph_complete = solver_start;
  auto optimization_complete = solver_start;
  try {
    auto options = rclcpp::NodeOptions();
    options.append_parameter_override("ceres_linear_solver", "SPARSE_NORMAL_CHOLESKY");
    options.append_parameter_override("ceres_preconditioner", "SCHUR_JACOBI");
    options.append_parameter_override("ceres_trust_strategy", "LEVENBERG_MARQUARDT");
    options.append_parameter_override("ceres_dogleg_type", "TRADITIONAL_DOGLEG");
    options.append_parameter_override("ceres_loss_function", "None");
    options.append_parameter_override("debug_logging", false);
    const auto node = std::make_shared<rclcpp::Node>("pose_graph_constraint_replay", options);
    node->declare_parameter<bool>("debug_logging", false);
    solver = solver_loader.createSharedInstance("solver_plugins::CeresSolver");
    solver->Configure(node);
    mapper.SetScanSolver(solver.get());

    std::cerr << "pose_graph_constraint_replay: rebuilding solver graph\n";
    for (const auto & sensor_vertices : graph->GetVertices()) {
      for (const auto & state_vertex : sensor_vertices.second) {
        if (state_vertex.second == nullptr) {
          throw std::runtime_error("existing graph contains a null vertex");
        }
        solver->AddNode(state_vertex.second);
      }
    }
    for (auto * edge : graph->GetEdges()) {
      solver->AddConstraint(edge);
    }

    for (auto & runtime : constraints) {
      const karto::Pose2 measurement(
        runtime.input.relative_x_m,
        runtime.input.relative_y_m,
        runtime.input.relative_yaw_rad);
      const karto::Pose2 expected_target = composePose(
        runtime.source->GetCorrectedPose(), measurement);
      const karto::Matrix3 local_covariance = covarianceMatrix(runtime.input.covariance);
      const karto::Matrix3 world_covariance = covarianceInWorldFrame(
        local_covariance, runtime.source->GetCorrectedPose().GetHeading());
      std::unique_ptr<karto::LinkInfo> link(new karto::LinkInfo(
          runtime.source->GetCorrectedPose(), expected_target, world_covariance));
      kt_bool is_new_edge = false;
      auto * edge = graph->AddEdge(runtime.source, runtime.target, is_new_edge);
      if (edge == nullptr || !is_new_edge) {
        throw std::runtime_error("edge creation failed after successful preflight");
      }
      edge->SetLabel(link.release());
      solver->AddConstraint(edge);
    }

    solver_graph_complete = std::chrono::steady_clock::now();
    std::cerr << "pose_graph_constraint_replay: optimizing grouped constraints\n";
    mapper.CorrectPoses();
    optimization_complete = std::chrono::steady_clock::now();
  } catch (const pluginlib::PluginlibException & exception) {
    std::cerr << "Solver plugin failure: " << exception.what() << '\n';
    return 11;
  } catch (const karto::Exception & exception) {
    std::cerr << "Karto optimization failure: " << exception.GetErrorMessage() << '\n';
    return 12;
  } catch (const std::exception & exception) {
    std::cerr << "Pose-graph optimization failure: " << exception.what() << '\n';
    return 13;
  }

  std::cerr << "pose_graph_constraint_replay: optimization complete\n";
  for (auto & runtime : constraints) {
    runtime.after = constraintResidual(runtime.source, runtime.target, runtime.input);
  }

  std::error_code filesystem_error;
  std::filesystem::create_directories(output_directory, filesystem_error);
  if (filesystem_error) {
    std::cerr << "Failed to create output directory: " << filesystem_error.message() << '\n';
    return 14;
  }
  if (
    !writeTrajectory(output_directory / "trajectory_before_after.csv", scans, before_poses) ||
    !writeConstraintDiagnostics(output_directory / "constraint_residuals.csv", constraints))
  {
    std::cerr << "Failed to write replay diagnostics\n";
    return 15;
  }

  if (!diagnostics_only) {
    if (!writeOccupancyMap(
        output_directory / "optimized_map", scans, resolution_m,
        min_pass_through, occupancy_threshold))
    {
      std::cerr << "Failed to write optimized occupancy map\n";
      return 15;
    }
    try {
      mapper.SaveToFile((output_directory / "optimized_pose_graph.posegraph").string());
    } catch (const karto::Exception & exception) {
      std::cerr << "Failed to save optimized pose graph: " <<
        exception.GetErrorMessage() << '\n';
      return 16;
    } catch (const std::exception & exception) {
      std::cerr << "Failed to save optimized pose graph: " << exception.what() << '\n';
      return 16;
    }
    std::filesystem::copy_file(
      input_data, output_directory / "optimized_pose_graph.data",
      std::filesystem::copy_options::overwrite_existing, filesystem_error);
    if (filesystem_error) {
      std::cerr << "Failed to copy companion graph data: " << filesystem_error.message() << '\n';
      return 17;
    }
  }
  const auto output_complete = std::chrono::steady_clock::now();
  const auto seconds = [](const auto begin, const auto end) {
      return std::chrono::duration<double>(end - begin).count();
    };

  std::ofstream summary(output_directory / "summary.csv");
  if (!summary.is_open()) {
    return 18;
  }
  summary << "scan_count,edge_count_before,constraint_count,edge_count_after,output_mode,"
          << "map_resolution_m,min_pass_through,occupancy_threshold,graph_load_s,"
          << "solver_rebuild_s,optimization_s,output_s\n"
          << scans.size() << ',' << graph->GetEdges().size() - constraints.size() << ','
          << constraints.size() << ',' << graph->GetEdges().size() << ','
          << (diagnostics_only ? "diagnostics_only" : "full") << ','
          << std::setprecision(17) << resolution_m << ',' << min_pass_through << ','
          << occupancy_threshold << ',' << seconds(load_start, load_complete) << ','
          << seconds(solver_start, solver_graph_complete) << ','
          << seconds(solver_graph_complete, optimization_complete) << ','
          << seconds(optimization_complete, output_complete) << '\n';
  std::cout << "Optimized " << scans.size() << " scans with " << constraints.size()
            << " grouped constraints; outputs: " << output_directory << '\n';
  return summary.good() ? 0 : 19;
}
