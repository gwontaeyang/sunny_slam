#ifndef SLAM_3D_LIDAR_INGEST__POINT_BUDGET_HPP_
#define SLAM_3D_LIDAR_INGEST__POINT_BUDGET_HPP_

#include <cstddef>
#include <cstdint>

namespace gole
{
namespace slam_3d_lidar_ingest
{

// One original point per ordered stratum; jitter avoids selecting only a fixed
// subset of rings when the input packet layout repeats every 32/128 points.
// The caller guarantees 2 <= count <= total and index < count.
inline std::size_t stratifiedPointIndex(
  const std::size_t index, const std::size_t count,
  const std::size_t total, const std::uint64_t seed)
{
  if (index == 0) {
    return 0;
  }
  if (index + 1 == count) {
    return total - 1;
  }
  const auto first = static_cast<std::uint64_t>(index) * total / count;
  const auto end = static_cast<std::uint64_t>(index + 1) * total / count;
  std::uint64_t hash = static_cast<std::uint64_t>(index) + seed * 0x9e3779b97f4a7c15ULL;
  hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
  hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
  hash ^= hash >> 31;
  return static_cast<std::size_t>(first + hash % (end - first));
}

}  // namespace slam_3d_lidar_ingest
}  // namespace gole

#endif  // SLAM_3D_LIDAR_INGEST__POINT_BUDGET_HPP_
