#ifndef SLAM_3D_LIVOX_INGEST__MID360_TIME_CONTRACT_HPP_
#define SLAM_3D_LIVOX_INGEST__MID360_TIME_CONTRACT_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace gole
{
namespace slam_3d_livox_ingest
{

enum class TimeContractCode
{
  kValid,
  kMissingTimebase,
  kTimebaseOutsideRosRange,
  kEmptyScan,
  kTooManyPoints,
  kPointCountMismatch,
  kInsufficientFinitePoints,
  kFirstOffsetTooLarge,
  kOffsetRegression,
  kZeroScanDuration,
  kScanDurationTooLong,
  kNonIncreasingTimebase,
};

struct TimeContractOptions
{
  std::size_t max_points{500000};
  std::uint64_t max_scan_duration_ns{200000000};
  std::uint32_t first_point_max_offset_ns{1000000};
  double min_finite_point_ratio{0.9};
  bool require_point_count_match{true};
  bool require_monotonic_offsets{true};
  bool require_zero_based_offset{true};
  bool require_nonzero_duration{true};
};

struct TimeContractResult
{
  TimeContractCode code{TimeContractCode::kValid};
  std::size_t declared_point_count{0};
  std::size_t actual_point_count{0};
  std::size_t finite_point_count{0};
  std::size_t invalid_point_count{0};
  std::size_t offset_regression_count{0};
  std::uint32_t first_offset_ns{0};
  std::uint32_t min_offset_ns{0};
  std::uint32_t max_offset_ns{0};
  std::uint64_t scan_duration_ns{0};
  double finite_point_ratio{0.0};

  bool accepted() const
  {
    return code == TimeContractCode::kValid;
  }
};

const char * timeContractCodeString(TimeContractCode code);

template<typename PointContainer>
TimeContractResult validateMid360TimeContract(
  const std::uint64_t timebase_ns,
  const std::uint32_t declared_point_count,
  const PointContainer & points,
  const TimeContractOptions & options)
{
  TimeContractResult result;
  result.declared_point_count = declared_point_count;
  result.actual_point_count = points.size();

  constexpr std::uint64_t kMaxRosTimeNs =
    static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) * 1000000000ULL +
    999999999ULL;

  if (timebase_ns == 0) {
    result.code = TimeContractCode::kMissingTimebase;
    return result;
  }
  if (timebase_ns > kMaxRosTimeNs) {
    result.code = TimeContractCode::kTimebaseOutsideRosRange;
    return result;
  }
  if (points.empty()) {
    result.code = TimeContractCode::kEmptyScan;
    return result;
  }
  if (points.size() > options.max_points) {
    result.code = TimeContractCode::kTooManyPoints;
    return result;
  }

  bool first = true;
  std::uint32_t previous_offset_ns = 0;
  result.min_offset_ns = std::numeric_limits<std::uint32_t>::max();
  for (const auto & point : points) {
    const std::uint32_t offset_ns = point.offset_time;
    if (first) {
      result.first_offset_ns = offset_ns;
      first = false;
    } else if (offset_ns < previous_offset_ns) {
      ++result.offset_regression_count;
    }
    previous_offset_ns = offset_ns;
    result.min_offset_ns = std::min(result.min_offset_ns, offset_ns);
    result.max_offset_ns = std::max(result.max_offset_ns, offset_ns);

    if (std::isfinite(static_cast<double>(point.x)) &&
      std::isfinite(static_cast<double>(point.y)) &&
      std::isfinite(static_cast<double>(point.z)))
    {
      ++result.finite_point_count;
    } else {
      ++result.invalid_point_count;
    }
  }

  result.scan_duration_ns =
    static_cast<std::uint64_t>(result.max_offset_ns) - result.min_offset_ns;
  result.finite_point_ratio =
    static_cast<double>(result.finite_point_count) /
    static_cast<double>(result.actual_point_count);

  if (options.require_point_count_match &&
    result.declared_point_count != result.actual_point_count)
  {
    result.code = TimeContractCode::kPointCountMismatch;
  } else if (
    result.finite_point_ratio <
    std::clamp(options.min_finite_point_ratio, 0.0, 1.0))
  {
    result.code = TimeContractCode::kInsufficientFinitePoints;
  } else if (
    options.require_zero_based_offset &&
    result.first_offset_ns > options.first_point_max_offset_ns)
  {
    result.code = TimeContractCode::kFirstOffsetTooLarge;
  } else if (
    options.require_monotonic_offsets &&
    result.offset_regression_count > 0)
  {
    result.code = TimeContractCode::kOffsetRegression;
  } else if (
    options.require_nonzero_duration &&
    result.actual_point_count > 1 &&
    result.scan_duration_ns == 0)
  {
    result.code = TimeContractCode::kZeroScanDuration;
  } else if (
    options.max_scan_duration_ns > 0 &&
    result.scan_duration_ns > options.max_scan_duration_ns)
  {
    result.code = TimeContractCode::kScanDurationTooLong;
  }

  return result;
}

}  // namespace slam_3d_livox_ingest
}  // namespace gole

#endif  // SLAM_3D_LIVOX_INGEST__MID360_TIME_CONTRACT_HPP_
