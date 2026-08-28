#include "slam_2d_core/roughness_loop_memory.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slam_2d_core
{
namespace
{

bool validSample(const RoughnessScanMemorySample & sample) noexcept
{
  if (sample.state_id < 0 || sample.unique_id < 0 || sample.stamp_ns <= 0) {
    return false;
  }
  if (
    sample.available &&
    (!std::isfinite(sample.memory_strength) ||
    sample.memory_strength < 0.0 || sample.memory_strength > 1.0 ||
    !std::isfinite(sample.effective_duration_s) ||
    sample.effective_duration_s < 0.0 ||
    !std::isfinite(sample.effective_distance_m) ||
    sample.effective_distance_m < 0.0))
  {
    return false;
  }
  return !sample.roughness_score_available ||
         (std::isfinite(sample.roughness_score) && sample.roughness_score >= 0.0);
}

double combinedStrength(const double source, const double target) noexcept
{
  return std::clamp(1.0 - (1.0 - source) * (1.0 - target), 0.0, 1.0);
}

}  // namespace

RoughnessLoopMemory::RoughnessLoopMemory(const std::size_t maximum_entries)
: maximum_entries_(maximum_entries)
{
  if (maximum_entries_ == 0U) {
    throw std::invalid_argument("maximum_entries must be positive");
  }
}

void RoughnessLoopMemory::reset(const std::uint64_t mapper_generation) noexcept
{
  mapper_generation_ = mapper_generation;
  insertion_order_.clear();
  samples_.clear();
}

bool RoughnessLoopMemory::record(
  const RoughnessScanMemorySample & sample) noexcept
{
  if (sample.mapper_generation != mapper_generation_ || !validSample(sample)) {
    return false;
  }

  const auto existing = samples_.find(sample.unique_id);
  if (existing != samples_.end()) {
    existing->second = sample;
    return true;
  }

  while (samples_.size() >= maximum_entries_) {
    const std::int32_t oldest_unique_id = insertion_order_.front();
    insertion_order_.pop_front();
    samples_.erase(oldest_unique_id);
  }
  insertion_order_.push_back(sample.unique_id);
  samples_.emplace(sample.unique_id, sample);
  return true;
}

const RoughnessScanMemorySample * RoughnessLoopMemory::find(
  const std::uint64_t mapper_generation,
  const std::int32_t unique_id) const noexcept
{
  if (mapper_generation != mapper_generation_ || unique_id < 0) {
    return nullptr;
  }
  const auto found = samples_.find(unique_id);
  return found == samples_.end() ? nullptr : &found->second;
}

RoughnessLoopMemoryPrior RoughnessLoopMemory::evaluate(
  const std::uint64_t mapper_generation,
  const std::vector<std::int32_t> & source_chain_unique_ids,
  const std::int32_t actual_source_unique_id,
  const std::int32_t target_unique_id) const noexcept
{
  RoughnessLoopMemoryPrior prior;
  prior.source_chain_size = source_chain_unique_ids.size();
  if (
    mapper_generation != mapper_generation_ ||
    source_chain_unique_ids.empty() || target_unique_id < 0)
  {
    return prior;
  }

  const auto * target = find(mapper_generation, target_unique_id);
  prior.target_available = target && target->available;
  if (prior.target_available) {
    prior.target_strength = target->memory_strength;
    if (target->roughness_score_available) {
      prior.target_roughness_score = target->roughness_score;
    }
  }

  const auto * actual_source = find(
    mapper_generation, actual_source_unique_id);
  prior.actual_source_available = actual_source && actual_source->available;
  if (prior.actual_source_available) {
    prior.actual_source_strength = actual_source->memory_strength;
  }

  for (const std::int32_t unique_id : source_chain_unique_ids) {
    const auto * source = find(mapper_generation, unique_id);
    if (!source || !source->available) {
      continue;
    }
    ++prior.source_available_count;
    if (
      prior.source_peak_unique_id < 0 ||
      source->memory_strength > prior.source_peak_strength)
    {
      prior.source_peak_unique_id = unique_id;
      prior.source_peak_strength = source->memory_strength;
    }
    if (
      source->roughness_score_available &&
      (!std::isfinite(prior.source_peak_roughness_score) ||
      source->roughness_score > prior.source_peak_roughness_score))
    {
      prior.source_peak_roughness_score = source->roughness_score;
    }
  }
  prior.source_coverage = static_cast<double>(prior.source_available_count) /
    static_cast<double>(prior.source_chain_size);
  prior.available = prior.target_available && prior.source_available_count > 0U;
  if (prior.available) {
    prior.combined_strength = combinedStrength(
      prior.source_peak_strength, prior.target_strength);
  }
  return prior;
}

}  // namespace slam_2d_core
