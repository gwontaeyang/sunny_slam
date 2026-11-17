#include "slam_2d_core/roughness_place_scan_chain.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace slam_2d_core
{
namespace
{

bool validScan(const RoughnessPlaceStableScan & scan) noexcept
{
  return
    !scan.sensor_name.empty() && scan.stamp_ns > 0 &&
    scan.state_id >= 0 && scan.unique_id >= 0 &&
    std::isfinite(scan.odom_x_m) && std::isfinite(scan.odom_y_m) &&
    std::isfinite(scan.odom_yaw_rad);
}

bool follows(
  const RoughnessPlaceStableScan & previous,
  const RoughnessPlaceStableScan & current) noexcept
{
  return
    previous.mapper_generation == current.mapper_generation &&
    previous.sensor_name == current.sensor_name &&
    current.stamp_ns > previous.stamp_ns &&
    current.state_id == previous.state_id + 1;
}

}  // namespace

bool validRoughnessPlaceScanChainConfig(
  const RoughnessPlaceScanChainConfig & config) noexcept
{
  return
    config.scan_count >= 2U &&
    std::isfinite(config.maximum_anchor_distance_m) &&
    config.maximum_anchor_distance_m > 0.0 &&
    std::isfinite(config.maximum_anchor_time_gap_s) &&
    config.maximum_anchor_time_gap_s > 0.0;
}

RoughnessPlaceScanChainTracker::RoughnessPlaceScanChainTracker(
  RoughnessPlaceScanChainConfig config)
{
  configure(config);
}

void RoughnessPlaceScanChainTracker::configure(
  const RoughnessPlaceScanChainConfig & config)
{
  if (!validRoughnessPlaceScanChainConfig(config)) {
    throw std::invalid_argument("Invalid roughness place scan-chain config");
  }
  config_ = config;
  reset();
}

void RoughnessPlaceScanChainTracker::reset()
{
  stable_history_.clear();
  last_complete_stable_chain_.clear();
  active_pre_chain_.clear();
  pending_events_.clear();
  last_stamp_ns_ = 0;
  mapper_generation_ = 0U;
  mapper_generation_initialized_ = false;
}

void RoughnessPlaceScanChainTracker::appendStableHistory(
  const RoughnessPlaceStableScan & scan)
{
  if (!stable_history_.empty() && !follows(stable_history_.back(), scan)) {
    stable_history_.clear();
  }
  stable_history_.push_back(scan);
  while (stable_history_.size() > config_.scan_count) {
    stable_history_.pop_front();
  }
  if (stable_history_.size() == config_.scan_count) {
    std::vector<RoughnessPlaceStableScan> candidate(
      stable_history_.begin(), stable_history_.end());
    if (chainWithinDistance(candidate, true)) {
      last_complete_stable_chain_ = std::move(candidate);
    }
  }
}

bool RoughnessPlaceScanChainTracker::chainWithinDistance(
  const std::vector<RoughnessPlaceStableScan> & chain,
  const bool anchor_is_last) const noexcept
{
  if (chain.size() != config_.scan_count) {
    return false;
  }
  const RoughnessPlaceStableScan & anchor =
    anchor_is_last ? chain.back() : chain.front();
  return std::all_of(
    chain.begin(), chain.end(),
    [this, &anchor](const RoughnessPlaceStableScan & scan) {
      return
      validScan(scan) && scan.strict_stable &&
      scan.mapper_generation == anchor.mapper_generation &&
      scan.sensor_name == anchor.sensor_name &&
      std::hypot(
        scan.odom_x_m - anchor.odom_x_m,
        scan.odom_y_m - anchor.odom_y_m) <=
      config_.maximum_anchor_distance_m;
    });
}

RoughnessPlaceScanChainEvent RoughnessPlaceScanChainTracker::complete(
  RoughnessPlaceScanChainEvent event,
  const char * failure_reason) const
{
  event.available = failure_reason == nullptr;
  event.failure_reason = failure_reason ? failure_reason : "none";
  return event;
}

const char * RoughnessPlaceScanChainTracker::postReacquisitionFailure(
  const RoughnessPlaceScanChainEvent & pending,
  const RoughnessPlaceStableScan & scan) const noexcept
{
  const std::int64_t event_state_gap =
    static_cast<std::int64_t>(scan.state_id) -
    static_cast<std::int64_t>(pending.event.post_anchor_state_id);
  if (
    event_state_gap < 0 ||
    static_cast<std::uint64_t>(event_state_gap) >
    config_.maximum_anchor_state_gap)
  {
    return "post_reacquisition_state_gap_exceeded";
  }
  const double event_time_gap_s = 1.0e-9 * static_cast<double>(
    scan.stamp_ns - pending.event.end_stamp_ns);
  if (
    !std::isfinite(event_time_gap_s) || event_time_gap_s < 0.0 ||
    event_time_gap_s > config_.maximum_anchor_time_gap_s)
  {
    return "post_reacquisition_time_gap_exceeded";
  }
  const double event_distance_m = std::hypot(
    scan.odom_x_m - pending.event.post_anchor.odom_x_m,
    scan.odom_y_m - pending.event.post_anchor.odom_y_m);
  if (
    !std::isfinite(event_distance_m) ||
    event_distance_m > config_.maximum_anchor_distance_m)
  {
    return "post_reacquisition_distance_exceeded";
  }
  return nullptr;
}

RoughnessPlaceScanChainUpdate RoughnessPlaceScanChainTracker::observe(
  const RoughnessPlaceStableScan & scan,
  const RoughnessPlaceTrackerAction tracker_action)
{
  RoughnessPlaceScanChainUpdate update;
  const bool timestamp_reset =
    tracker_action == RoughnessPlaceTrackerAction::kTimestampReset ||
    (last_stamp_ns_ != 0 && scan.stamp_ns < last_stamp_ns_);
  const bool generation_reset =
    mapper_generation_initialized_ &&
    scan.mapper_generation != mapper_generation_;
  if (timestamp_reset || generation_reset) {
    for (RoughnessPlaceScanChainEvent & pending : pending_events_) {
      update.completed_events.push_back(
        complete(
          std::move(pending),
          timestamp_reset ? "timestamp_reset" : "mapper_generation_reset"));
    }
    reset();
    return update;
  }
  if (last_stamp_ns_ != 0 && scan.stamp_ns == last_stamp_ns_) {
    return update;
  }
  if (!validScan(scan)) {
    for (RoughnessPlaceScanChainEvent & pending : pending_events_) {
      update.completed_events.push_back(
        complete(std::move(pending), "post_scan_invalid"));
    }
    pending_events_.clear();
    stable_history_.clear();
    last_complete_stable_chain_.clear();
    active_pre_chain_.clear();
    return update;
  }
  last_stamp_ns_ = scan.stamp_ns;
  mapper_generation_ = scan.mapper_generation;
  mapper_generation_initialized_ = true;

  if (!scan.quality_valid) {
    for (RoughnessPlaceScanChainEvent & pending : pending_events_) {
      update.completed_events.push_back(
        complete(std::move(pending), "post_quality_unavailable"));
    }
    pending_events_.clear();
    stable_history_.clear();
    last_complete_stable_chain_.clear();
    active_pre_chain_.clear();
    return update;
  }

  std::vector<RoughnessPlaceScanChainEvent> still_pending;
  still_pending.reserve(pending_events_.size());
  for (RoughnessPlaceScanChainEvent & pending : pending_events_) {
    if (
      !pending.post_chain.empty() &&
      (!scan.strict_stable || !follows(pending.post_chain.back(), scan)))
    {
      pending.post_chain.clear();
      ++pending.post_reacquisition_reset_count;
    }
    if (pending.post_chain.empty()) {
      const char * failure_reason = postReacquisitionFailure(pending, scan);
      if (failure_reason) {
        update.completed_events.push_back(
          complete(std::move(pending), failure_reason));
        continue;
      }
    }
    if (!scan.strict_stable) {
      still_pending.push_back(std::move(pending));
      continue;
    }
    pending.post_chain.push_back(scan);
    if (pending.post_chain.size() < config_.scan_count) {
      still_pending.push_back(std::move(pending));
      continue;
    }
    if (chainWithinDistance(pending.post_chain, false)) {
      update.completed_events.push_back(complete(std::move(pending), nullptr));
      continue;
    }
    pending.post_chain.erase(pending.post_chain.begin());
    ++pending.post_reacquisition_reset_count;
    still_pending.push_back(std::move(pending));
  }
  pending_events_ = std::move(still_pending);

  if (tracker_action == RoughnessPlaceTrackerAction::kStarted) {
    if (stable_history_.size() == config_.scan_count) {
      active_pre_chain_.assign(stable_history_.begin(), stable_history_.end());
    } else {
      active_pre_chain_ = last_complete_stable_chain_;
    }
  }
  if (
    scan.strict_stable &&
    tracker_action != RoughnessPlaceTrackerAction::kStarted &&
    tracker_action != RoughnessPlaceTrackerAction::kExtended)
  {
    appendStableHistory(scan);
  } else {
    stable_history_.clear();
  }
  return update;
}

RoughnessPlaceScanChainUpdate RoughnessPlaceScanChainTracker::finalizeEvent(
  const RoughnessPlaceEvent & event,
  std::vector<RoughnessPlaceCandidate> candidates,
  const RoughnessPlaceStableScan & post_anchor)
{
  RoughnessPlaceScanChainUpdate update;
  RoughnessPlaceScanChainEvent pending;
  pending.event = event;
  pending.candidates = std::move(candidates);
  pending.pre_chain = std::move(active_pre_chain_);
  active_pre_chain_.clear();

  if (!validScan(post_anchor)) {
    update.completed_events.push_back(
      complete(std::move(pending), "post_anchor_invalid"));
    return update;
  }
  if (!post_anchor.quality_valid) {
    update.completed_events.push_back(
      complete(std::move(pending), "post_anchor_quality_unavailable"));
    return update;
  }
  if (pending.pre_chain.empty()) {
    update.completed_events.push_back(
      complete(std::move(pending), "pre_chain_unavailable"));
    return update;
  }
  if (
    post_anchor.state_id != event.post_anchor_state_id ||
    post_anchor.unique_id != event.post_anchor.scan_unique_id)
  {
    update.completed_events.push_back(
      complete(std::move(pending), "post_anchor_identity_mismatch"));
    return update;
  }
  if (!chainWithinDistance(pending.pre_chain, true)) {
    update.completed_events.push_back(
      complete(std::move(pending), "pre_chain_incomplete_or_too_distant"));
    return update;
  }
  const RoughnessPlaceStableScan & pre_chain_anchor = pending.pre_chain.back();
  const std::int64_t pre_anchor_state_gap =
    static_cast<std::int64_t>(event.pre_anchor_state_id) -
    static_cast<std::int64_t>(pre_chain_anchor.state_id);
  if (pre_anchor_state_gap < 0) {
    update.completed_events.push_back(
      complete(std::move(pending), "pre_chain_anchor_order_mismatch"));
    return update;
  }
  if (
    static_cast<std::uint64_t>(pre_anchor_state_gap) >
    config_.maximum_anchor_state_gap)
  {
    update.completed_events.push_back(
      complete(std::move(pending), "pre_chain_anchor_state_gap_exceeded"));
    return update;
  }
  const double pre_anchor_time_gap_s = 1.0e-9 * static_cast<double>(
    event.pre_anchor.stamp_ns - pre_chain_anchor.stamp_ns);
  if (
    !std::isfinite(pre_anchor_time_gap_s) || pre_anchor_time_gap_s < 0.0 ||
    pre_anchor_time_gap_s > config_.maximum_anchor_time_gap_s)
  {
    update.completed_events.push_back(
      complete(std::move(pending), "pre_chain_anchor_time_gap_exceeded"));
    return update;
  }
  const double pre_anchor_distance_m = std::hypot(
    pre_chain_anchor.odom_x_m - event.pre_anchor.odom_x_m,
    pre_chain_anchor.odom_y_m - event.pre_anchor.odom_y_m);
  if (
    !std::isfinite(pre_anchor_distance_m) ||
    pre_anchor_distance_m > config_.maximum_anchor_distance_m)
  {
    update.completed_events.push_back(
      complete(std::move(pending), "pre_chain_anchor_distance_exceeded"));
    return update;
  }
  if (post_anchor.strict_stable) {
    pending.post_chain.push_back(post_anchor);
  } else {
    ++pending.post_reacquisition_reset_count;
  }
  pending_events_.push_back(std::move(pending));
  return update;
}

}  // namespace slam_2d_core
