#include "slam_3d_livox_ingest/mid360_time_contract.hpp"

namespace gole
{
namespace slam_3d_livox_ingest
{

const char * timeContractCodeString(const TimeContractCode code)
{
  switch (code) {
    case TimeContractCode::kValid:
      return "valid";
    case TimeContractCode::kMissingTimebase:
      return "missing_timebase";
    case TimeContractCode::kTimebaseOutsideRosRange:
      return "timebase_outside_ros_range";
    case TimeContractCode::kEmptyScan:
      return "empty_scan";
    case TimeContractCode::kTooManyPoints:
      return "too_many_points";
    case TimeContractCode::kPointCountMismatch:
      return "point_count_mismatch";
    case TimeContractCode::kInsufficientFinitePoints:
      return "insufficient_finite_points";
    case TimeContractCode::kFirstOffsetTooLarge:
      return "first_offset_too_large";
    case TimeContractCode::kOffsetRegression:
      return "offset_regression";
    case TimeContractCode::kZeroScanDuration:
      return "zero_scan_duration";
    case TimeContractCode::kScanDurationTooLong:
      return "scan_duration_too_long";
    case TimeContractCode::kNonIncreasingTimebase:
      return "non_increasing_timebase";
  }
  return "unknown";
}

}  // namespace slam_3d_livox_ingest
}  // namespace gole
