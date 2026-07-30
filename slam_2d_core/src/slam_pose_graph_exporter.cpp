#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "karto_sdk/Mapper.h"

namespace
{

void printUsage(const char * executable)
{
  std::cerr << "Usage: " << executable
            << " <pose-graph-stem> <output-csv>\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 3) {
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
  return exported_count > 0 ? 0 : 6;
}
