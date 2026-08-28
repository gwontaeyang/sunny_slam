#ifndef SLAM_2D_CORE__ROUGHNESS_LOOP_MEMORY_HPP_
#define SLAM_2D_CORE__ROUGHNESS_LOOP_MEMORY_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <vector>

namespace slam_2d_core
{

struct RoughnessScanMemorySample
{
  std::uint64_t mapper_generation{0};
  std::int32_t state_id{-1};
  std::int32_t unique_id{-1};
  std::int64_t stamp_ns{0};
  bool available{false};
  bool rough_trigger{false};
  bool roughness_score_available{false};
  double roughness_score{std::numeric_limits<double>::quiet_NaN()};
  double memory_strength{0.0};
  double effective_duration_s{0.0};
  double effective_distance_m{0.0};
};

struct RoughnessLoopMemoryPrior
{
  bool available{false};
  bool target_available{false};
  bool actual_source_available{false};
  std::size_t source_chain_size{0};
  std::size_t source_available_count{0};
  std::int32_t source_peak_unique_id{-1};
  double source_coverage{0.0};
  double source_peak_strength{0.0};
  double actual_source_strength{0.0};
  double target_strength{0.0};
  double combined_strength{0.0};
  double source_peak_roughness_score{
    std::numeric_limits<double>::quiet_NaN()};
  double target_roughness_score{std::numeric_limits<double>::quiet_NaN()};
};

class RoughnessLoopMemory
{
public:
  explicit RoughnessLoopMemory(std::size_t maximum_entries = 100000U);

  void reset(std::uint64_t mapper_generation) noexcept;
  bool record(const RoughnessScanMemorySample & sample) noexcept;
  const RoughnessScanMemorySample * find(
    std::uint64_t mapper_generation,
    std::int32_t unique_id) const noexcept;
  RoughnessLoopMemoryPrior evaluate(
    std::uint64_t mapper_generation,
    const std::vector<std::int32_t> & source_chain_unique_ids,
    std::int32_t actual_source_unique_id,
    std::int32_t target_unique_id) const noexcept;

  std::uint64_t mapperGeneration() const noexcept {return mapper_generation_;}
  std::size_t size() const noexcept {return samples_.size();}
  std::size_t maximumEntries() const noexcept {return maximum_entries_;}

private:
  std::size_t maximum_entries_{100000U};
  std::uint64_t mapper_generation_{0};
  std::deque<std::int32_t> insertion_order_;
  std::unordered_map<std::int32_t, RoughnessScanMemorySample> samples_;
};

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__ROUGHNESS_LOOP_MEMORY_HPP_
