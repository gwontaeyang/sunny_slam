#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "karto_sdk/Mapper.h"

namespace
{

void printUsage(const char * executable)
{
  std::cerr << "Usage: " << executable
            << " <pose-graph-stem> <node-output-csv> <edge-output-csv>\n";
}

double normalizedAngle(const double angle_rad)
{
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

karto::Pose2 relativePose(
  const karto::Pose2 & source,
  const karto::Pose2 & target)
{
  karto::Transform transform(source, karto::Pose2());
  return transform.TransformPose(target);
}

double covarianceWeightedSquared(
  const double x,
  const double y,
  const double yaw,
  const karto::Matrix3 & covariance)
{
  karto::Matrix3 regularized = covariance;
  regularized(0, 0) += 1.0e-9;
  regularized(1, 1) += 1.0e-9;
  regularized(2, 2) += 1.0e-9;
  karto::Matrix3 information;
  if (!regularized.InverseFast(information, 1.0e-14)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double residual[3] = {x, y, yaw};
  double value = 0.0;
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      value += residual[row] * information(row, column) * residual[column];
    }
  }
  return std::isfinite(value) ? value :
         std::numeric_limits<double>::quiet_NaN();
}

bool exportEdges(karto::Mapper & mapper, const std::string & output_csv)
{
  std::ofstream output(output_csv);
  if (!output.is_open()) {
    std::cerr << "Failed to open edge output CSV '" << output_csv << "'\n";
    return false;
  }
  output << "edge_index,endpoint_adjacency_class,source_sensor,target_sensor,"
         << "source_stamp_s,target_stamp_s,source_state_id,target_state_id,"
         << "source_unique_id,target_unique_id,relative_x_m,relative_y_m,"
         << "relative_yaw_rad,covariance_xx,covariance_xy,covariance_xyaw,"
         << "covariance_yy,covariance_yyaw,covariance_yawyaw,"
         << "final_expected_relative_x_m,final_expected_relative_y_m,"
         << "final_expected_relative_yaw_rad,final_residual_x_m,"
         << "final_residual_y_m,final_residual_yaw_rad,"
         << "final_residual_translation_m,"
         << "final_covariance_weighted_residual_sq\n";
  output << std::setprecision(17);

  std::size_t edge_index = 0;
  std::size_t exported_count = 0;
  auto * graph = mapper.GetGraph();
  if (!graph) {
    std::cerr << "Loaded pose graph has no graph object\n";
    return false;
  }
  for (auto * edge : graph->GetEdges()) {
    const std::size_t current_index = edge_index++;
    if (
      edge == nullptr || edge->GetSource() == nullptr ||
      edge->GetTarget() == nullptr ||
      edge->GetSource()->GetObject() == nullptr ||
      edge->GetTarget()->GetObject() == nullptr)
    {
      continue;
    }
    auto * link = dynamic_cast<karto::LinkInfo *>(edge->GetLabel());
    if (link == nullptr) {
      continue;
    }
    const auto * source = edge->GetSource()->GetObject();
    const auto * target = edge->GetTarget()->GetObject();
    const karto::Pose2 & measured = link->GetPoseDifference();
    const karto::Matrix3 & covariance = link->GetCovariance();
    const karto::Pose2 expected = relativePose(
      source->GetCorrectedPose(), target->GetCorrectedPose());
    const double residual_x = expected.GetX() - measured.GetX();
    const double residual_y = expected.GetY() - measured.GetY();
    const double residual_yaw = normalizedAngle(
      expected.GetHeading() - measured.GetHeading());
    const double weighted_residual = covarianceWeightedSquared(
      residual_x, residual_y, residual_yaw, covariance);
    const bool adjacent =
      source->GetSensorName() == target->GetSensorName() &&
      std::abs(source->GetStateId() - target->GetStateId()) <= 1;

    output << current_index << ','
           << (adjacent ? "same_sensor_adjacent_state" :
    "nonadjacent_or_cross_sensor") << ','
           << source->GetSensorName().ToString() << ','
           << target->GetSensorName().ToString() << ','
           << source->GetTime() << ','
           << target->GetTime() << ','
           << source->GetStateId() << ','
           << target->GetStateId() << ','
           << source->GetUniqueId() << ','
           << target->GetUniqueId() << ','
           << measured.GetX() << ','
           << measured.GetY() << ','
           << measured.GetHeading() << ','
           << covariance(0, 0) << ','
           << covariance(0, 1) << ','
           << covariance(0, 2) << ','
           << covariance(1, 1) << ','
           << covariance(1, 2) << ','
           << covariance(2, 2) << ','
           << expected.GetX() << ','
           << expected.GetY() << ','
           << expected.GetHeading() << ','
           << residual_x << ','
           << residual_y << ','
           << residual_yaw << ','
           << std::hypot(residual_x, residual_y) << ','
           << weighted_residual << '\n';
    ++exported_count;
  }
  if (!output.good()) {
    std::cerr << "Failed while writing edge output CSV '" << output_csv << "'\n";
    return false;
  }
  std::cout << "Exported " << exported_count
            << " final pose-graph edges to " << output_csv << '\n';
  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 4) {
    printUsage(argv[0]);
    return 2;
  }

  const std::string pose_graph_stem = argv[1];
  const std::string output_csv = argv[2];
  karto::Mapper mapper;

  try {
    mapper.LoadFromFile(pose_graph_stem + ".posegraph");
  } catch (karto::Exception & exception) {
    std::cerr << "Failed to load pose graph '" << pose_graph_stem
              << "': " << exception.GetErrorMessage() << '\n';
    return 3;
  } catch (const std::exception & exception) {
    std::cerr << "Failed to load pose graph '" << pose_graph_stem
              << "': " << exception.what() << '\n';
    return 3;
  }

  karto::LocalizedRangeScanVector scans = mapper.GetAllProcessedScans();
  std::sort(
    scans.begin(), scans.end(),
    [](const karto::LocalizedRangeScan * lhs, const karto::LocalizedRangeScan * rhs) {
      if (lhs->GetTime() == rhs->GetTime()) {
        return lhs->GetStateId() < rhs->GetStateId();
      }
      return lhs->GetTime() < rhs->GetTime();
    });

  std::ofstream output(output_csv);
  if (!output.is_open()) {
    std::cerr << "Failed to open output CSV '" << output_csv << "'\n";
    return 4;
  }

  output << "stamp_s,x_m,y_m,yaw_rad,source,state_id,unique_id,"
         << "odom_x_m,odom_y_m,odom_yaw_rad\n";
  output << std::setprecision(17);
  std::size_t exported_count = 0;
  for (const karto::LocalizedRangeScan * scan : scans) {
    if (scan == nullptr) {
      continue;
    }
    const karto::Pose2 & corrected = scan->GetCorrectedPose();
    const karto::Pose2 & odometric = scan->GetOdometricPose();
    if (!std::isfinite(scan->GetTime()) ||
      !std::isfinite(corrected.GetX()) ||
      !std::isfinite(corrected.GetY()) ||
      !std::isfinite(corrected.GetHeading()))
    {
      continue;
    }
    output << scan->GetTime() << ','
           << corrected.GetX() << ','
           << corrected.GetY() << ','
           << corrected.GetHeading() << ','
           << "pose_graph_final" << ','
           << scan->GetStateId() << ','
           << scan->GetUniqueId() << ','
           << odometric.GetX() << ','
           << odometric.GetY() << ','
           << odometric.GetHeading() << '\n';
    ++exported_count;
  }

  if (!output.good()) {
    std::cerr << "Failed while writing output CSV '" << output_csv << "'\n";
    return 5;
  }

  std::cout << "Exported " << exported_count
            << " final pose-graph scans to " << output_csv << '\n';
  if (!exportEdges(mapper, argv[3])) {
    return 7;
  }
  return exported_count > 0 ? 0 : 6;
}
