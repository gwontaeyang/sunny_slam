#ifndef SLAM_3D_CORE__SCAN_CONTEXT_DESCRIPTOR_HPP_
#define SLAM_3D_CORE__SCAN_CONTEXT_DESCRIPTOR_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace gole
{
namespace slam_3d_core
{

struct ScanContextMatch
{
  double distance{std::numeric_limits<double>::infinity()};
  int best_shift{0};
  int compared_sectors{0};
};

inline ScanContextMatch matchScanContextDescriptors(
  const std::vector<float> & lhs,
  const std::vector<float> & rhs,
  const int sectors)
{
  ScanContextMatch best;
  if (sectors <= 0 || lhs.empty() || lhs.size() != rhs.size() ||
    lhs.size() % static_cast<std::size_t>(sectors) != 0U)
  {
    return best;
  }

  const int rings = static_cast<int>(lhs.size() / static_cast<std::size_t>(sectors));
  constexpr double kMinimumNorm = 1.0e-12;
  for (int shift = 0; shift < sectors; ++shift) {
    double cosine_similarity_sum = 0.0;
    int compared_sectors = 0;
    for (int sector = 0; sector < sectors; ++sector) {
      double dot = 0.0;
      double lhs_squared_norm = 0.0;
      double rhs_squared_norm = 0.0;
      for (int ring = 0; ring < rings; ++ring) {
        const int shifted_sector = (sector + shift) % sectors;
        const double lhs_value = static_cast<double>(
          lhs[static_cast<std::size_t>(ring * sectors + shifted_sector)]);
        const double rhs_value = static_cast<double>(
          rhs[static_cast<std::size_t>(ring * sectors + sector)]);
        if (!std::isfinite(lhs_value) || !std::isfinite(rhs_value)) {
          continue;
        }
        dot += lhs_value * rhs_value;
        lhs_squared_norm += lhs_value * lhs_value;
        rhs_squared_norm += rhs_value * rhs_value;
      }
      if (lhs_squared_norm <= kMinimumNorm || rhs_squared_norm <= kMinimumNorm) {
        continue;
      }
      const double cosine_similarity = std::clamp(
        dot / std::sqrt(lhs_squared_norm * rhs_squared_norm),
        -1.0,
        1.0);
      cosine_similarity_sum += cosine_similarity;
      ++compared_sectors;
    }
    if (compared_sectors == 0) {
      continue;
    }
    const double distance = 1.0 -
      cosine_similarity_sum / static_cast<double>(compared_sectors);
    if (distance < best.distance) {
      best.distance = distance;
      best.best_shift = shift;
      best.compared_sectors = compared_sectors;
    }
  }

  return best;
}

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__SCAN_CONTEXT_DESCRIPTOR_HPP_
