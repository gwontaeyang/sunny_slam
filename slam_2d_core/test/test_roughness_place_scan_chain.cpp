#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "slam_2d_core/roughness_place_scan_chain.hpp"

namespace slam_2d_core
{
namespace
{

RoughnessPlaceStableScan scan(
  const std::int32_t state_id,
  const bool strict_stable,
  const double x_m = 0.1)
{
  RoughnessPlaceStableScan value;
  value.mapper_generation = 1U;
  value.sensor_name = "laser";
  value.stamp_ns = static_cast<std::int64_t>(state_id + 1) * 1'000'000'000LL;
  value.state_id = state_id;
  value.unique_id = state_id + 100;
  value.odom_x_m = x_m * static_cast<double>(state_id);
  value.odom_y_m = 0.0;
  value.odom_yaw_rad = 0.0;
  value.quality_valid = true;
  value.strict_stable = strict_stable;
  return value;
}

RoughnessPlaceEvent event(
  const std::int32_t pre_state_id,
  const std::int32_t post_state_id)
{
  RoughnessPlaceEvent value;
  value.event_id = 7U;
  value.start_stamp_ns =
    static_cast<std::int64_t>(pre_state_id + 1) * 1'000'000'000LL;
  value.end_stamp_ns =
    static_cast<std::int64_t>(post_state_id + 1) * 1'000'000'000LL;
  value.pre_anchor_state_id = pre_state_id;
  value.post_anchor_state_id = post_state_id;
  value.pre_anchor.stamp_ns = value.start_stamp_ns;
  value.pre_anchor.scan_state_id = pre_state_id;
  value.pre_anchor.scan_unique_id = pre_state_id + 100;
  value.pre_anchor.odom_x_m = 0.1 * static_cast<double>(pre_state_id);
  value.pre_anchor.odom_y_m = 0.0;
  value.pre_anchor.odom_heading_rad = 0.0;
  value.post_anchor.stamp_ns = value.end_stamp_ns;
  value.post_anchor.scan_state_id = post_state_id;
  value.post_anchor.scan_unique_id = post_state_id + 100;
  value.post_anchor.odom_x_m = 0.1 * static_cast<double>(post_state_id);
  value.post_anchor.odom_y_m = 0.0;
  value.post_anchor.odom_heading_rad = 0.0;
  return value;
}

RoughnessPlaceScanChainTracker primedTracker(
  const bool post_anchor_strict_stable = true)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0});
  EXPECT_TRUE(
    tracker.observe(
      scan(1, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.observe(
      scan(2, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.observe(
      scan(3, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.observe(
      scan(4, false), RoughnessPlaceTrackerAction::kStarted).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.observe(
      scan(5, false), RoughnessPlaceTrackerAction::kExtended).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.observe(
      scan(6, post_anchor_strict_stable),
      RoughnessPlaceTrackerAction::kFinalized).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.finalizeEvent(
      event(3, 6), {}, scan(6, post_anchor_strict_stable)).
    completed_events.empty());
  return tracker;
}

TEST(RoughnessPlaceScanChainTracker, WaitsForCompleteStablePostChain)
{
  auto tracker = primedTracker();
  EXPECT_TRUE(
    tracker.observe(
      scan(7, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated).
    completed_events.empty());

  const auto update = tracker.observe(
    scan(8, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  const auto & completed = update.completed_events.front();
  EXPECT_TRUE(completed.available);
  EXPECT_EQ(completed.failure_reason, "none");
  ASSERT_EQ(completed.pre_chain.size(), 3U);
  ASSERT_EQ(completed.post_chain.size(), 3U);
  EXPECT_EQ(completed.pre_chain.front().state_id, 1);
  EXPECT_EQ(completed.pre_chain.back().state_id, 3);
  EXPECT_EQ(completed.post_chain.front().state_id, 6);
  EXPECT_EQ(completed.post_chain.back().state_id, 8);
}

TEST(RoughnessPlaceScanChainTracker, ReacquiresAfterUnstablePostScan)
{
  auto tracker = primedTracker();
  EXPECT_TRUE(
    tracker.observe(
      scan(7, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.observe(scan(8, false), RoughnessPlaceTrackerAction::kStarted).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.observe(
      scan(9, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated).
    completed_events.empty());
  EXPECT_TRUE(
    tracker.observe(
      scan(10, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated).
    completed_events.empty());
  const auto update = tracker.observe(
    scan(11, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  const auto & completed = update.completed_events.front();
  EXPECT_TRUE(completed.available);
  EXPECT_EQ(completed.failure_reason, "none");
  EXPECT_EQ(completed.post_chain.front().state_id, 9);
  EXPECT_EQ(completed.post_chain.back().state_id, 11);
  EXPECT_EQ(completed.post_reacquisition_reset_count, 1U);
}

TEST(RoughnessPlaceScanChainTracker, MissingPostQualityFailsOpen)
{
  auto tracker = primedTracker();
  auto unavailable = scan(7, false);
  unavailable.quality_valid = false;

  const auto update = tracker.observe(
    unavailable, RoughnessPlaceTrackerAction::kIgnoredInvalid);

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "post_quality_unavailable");
}

TEST(RoughnessPlaceScanChainTracker, UsesLastCompleteStablePreChain)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0});
  tracker.observe(scan(1, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(2, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(3, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(4, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(5, false), RoughnessPlaceTrackerAction::kStarted);
  tracker.observe(scan(6, true), RoughnessPlaceTrackerAction::kFinalized);
  tracker.finalizeEvent(event(4, 6), {}, scan(6, true));
  tracker.observe(scan(7, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  const auto update = tracker.observe(
    scan(8, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  const auto & completed = update.completed_events.front();
  EXPECT_TRUE(completed.available);
  ASSERT_EQ(completed.pre_chain.size(), 3U);
  EXPECT_EQ(completed.pre_chain.front().state_id, 1);
  EXPECT_EQ(completed.pre_chain.back().state_id, 3);
  EXPECT_EQ(completed.event.pre_anchor_state_id, 4);
}

TEST(RoughnessPlaceScanChainTracker, RejectsStalePreChainByStateGap)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0, 0U, 5.0});
  tracker.observe(scan(1, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(2, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(3, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(4, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(5, false), RoughnessPlaceTrackerAction::kStarted);
  tracker.observe(scan(6, true), RoughnessPlaceTrackerAction::kFinalized);

  const auto update = tracker.finalizeEvent(event(4, 6), {}, scan(6, true));

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "pre_chain_anchor_state_gap_exceeded");
}

TEST(RoughnessPlaceScanChainTracker, RejectsStalePreChainByTimeGap)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0, 5U, 0.5});
  tracker.observe(scan(1, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(2, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(3, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(4, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(5, false), RoughnessPlaceTrackerAction::kStarted);
  tracker.observe(scan(6, true), RoughnessPlaceTrackerAction::kFinalized);

  const auto update = tracker.finalizeEvent(event(4, 6), {}, scan(6, true));

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "pre_chain_anchor_time_gap_exceeded");
}

TEST(RoughnessPlaceScanChainTracker, BoundsPostReacquisitionByDistance)
{
  auto tracker = primedTracker(false);
  auto distant = scan(7, true);
  distant.odom_x_m = 2.0;

  const auto update = tracker.observe(
    distant, RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "post_reacquisition_distance_exceeded");
}

TEST(RoughnessPlaceScanChainTracker, BoundsPostReacquisitionByStateGap)
{
  auto tracker = primedTracker(false);

  tracker.observe(
    scan(7, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(
    scan(8, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(
    scan(9, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(
    scan(10, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(
    scan(11, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  const auto update = tracker.observe(
    scan(12, false), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "post_reacquisition_state_gap_exceeded");
}

TEST(RoughnessPlaceScanChainTracker, BoundsPostReacquisitionByTimeGap)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0, 5U, 0.5});
  tracker.observe(scan(1, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(2, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(3, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(4, false), RoughnessPlaceTrackerAction::kStarted);
  tracker.observe(scan(5, false), RoughnessPlaceTrackerAction::kExtended);
  tracker.observe(scan(6, false), RoughnessPlaceTrackerAction::kFinalized);
  tracker.finalizeEvent(event(3, 6), {}, scan(6, false));

  const auto update = tracker.observe(
    scan(7, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "post_reacquisition_time_gap_exceeded");
}

TEST(RoughnessPlaceScanChainTracker, BoundsOnlyStableReacquisitionStart)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0, 0U, 0.5});
  tracker.observe(scan(1, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(2, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(3, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(4, false), RoughnessPlaceTrackerAction::kStarted);
  tracker.observe(scan(5, false), RoughnessPlaceTrackerAction::kExtended);
  tracker.observe(scan(6, true), RoughnessPlaceTrackerAction::kFinalized);
  tracker.finalizeEvent(event(3, 6), {}, scan(6, true));

  EXPECT_TRUE(
    tracker.observe(
      scan(7, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated).
    completed_events.empty());
  const auto update = tracker.observe(
    scan(8, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_TRUE(update.completed_events.front().available);
  EXPECT_EQ(update.completed_events.front().post_chain.front().state_id, 6);
  EXPECT_EQ(update.completed_events.front().post_chain.back().state_id, 8);
}

TEST(RoughnessPlaceScanChainTracker, DuplicateTimestampIsIgnored)
{
  auto tracker = primedTracker();
  const auto duplicate = tracker.observe(
    scan(6, true), RoughnessPlaceTrackerAction::kIgnoredInvalid);
  EXPECT_TRUE(duplicate.completed_events.empty());
  tracker.observe(
    scan(7, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  const auto update = tracker.observe(
    scan(8, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_TRUE(update.completed_events.front().available);
}

TEST(RoughnessPlaceScanChainTracker, TimestampResetReportsPendingFailOpen)
{
  auto tracker = primedTracker();
  auto rollback = scan(0, false);
  rollback.stamp_ns = 1;

  const auto update = tracker.observe(
    rollback, RoughnessPlaceTrackerAction::kTimestampReset);

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(update.completed_events.front().failure_reason, "timestamp_reset");
}

TEST(RoughnessPlaceScanChainTracker, RejectsIncompletePreChainAfterRollback)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0});
  tracker.observe(scan(1, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(2, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  auto rollback = scan(0, false);
  rollback.stamp_ns = 1;
  tracker.observe(rollback, RoughnessPlaceTrackerAction::kTimestampReset);
  tracker.observe(scan(4, false), RoughnessPlaceTrackerAction::kStarted);
  tracker.observe(scan(5, false), RoughnessPlaceTrackerAction::kExtended);
  tracker.observe(scan(6, true), RoughnessPlaceTrackerAction::kFinalized);

  const auto update = tracker.finalizeEvent(event(3, 6), {}, scan(6, true));

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "pre_chain_unavailable");
}

TEST(RoughnessPlaceScanChainTracker, RejectsPreChainBeyondDistanceLimit)
{
  RoughnessPlaceScanChainTracker tracker({3U, 0.5});
  tracker.observe(scan(1, true, 0.4), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(2, true, 0.4), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(3, true, 0.4), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(4, false), RoughnessPlaceTrackerAction::kStarted);
  tracker.observe(scan(5, false), RoughnessPlaceTrackerAction::kExtended);
  tracker.observe(scan(6, true), RoughnessPlaceTrackerAction::kFinalized);

  const auto update = tracker.finalizeEvent(event(3, 6), {}, scan(6, true));

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "pre_chain_incomplete_or_too_distant");
}

TEST(RoughnessPlaceScanChainTracker, MapperGenerationChangeResetsHistory)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0});
  tracker.observe(scan(1, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  tracker.observe(scan(2, true), RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  auto changed = scan(3, true);
  changed.mapper_generation = 2U;

  const auto update = tracker.observe(
    changed, RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  EXPECT_TRUE(update.completed_events.empty());
  auto started = scan(4, false);
  started.mapper_generation = 2U;
  tracker.observe(started, RoughnessPlaceTrackerAction::kStarted);
  auto extended = scan(5, false);
  extended.mapper_generation = 2U;
  tracker.observe(extended, RoughnessPlaceTrackerAction::kExtended);
  auto post = scan(6, true);
  post.mapper_generation = 2U;
  tracker.observe(post, RoughnessPlaceTrackerAction::kFinalized);
  const auto finalized = tracker.finalizeEvent(event(3, 6), {}, post);
  ASSERT_EQ(finalized.completed_events.size(), 1U);
  EXPECT_FALSE(finalized.completed_events.front().available);
}

TEST(RoughnessPlaceScanChainTracker, GenerationChangeFailsOpenPendingEvent)
{
  auto tracker = primedTracker();
  auto changed = scan(7, true);
  changed.mapper_generation = 2U;

  const auto update = tracker.observe(
    changed, RoughnessPlaceTrackerAction::kStableAnchorUpdated);

  ASSERT_EQ(update.completed_events.size(), 1U);
  EXPECT_FALSE(update.completed_events.front().available);
  EXPECT_EQ(
    update.completed_events.front().failure_reason,
    "mapper_generation_reset");
}

TEST(RoughnessPlaceScanChainTracker, ZeroMapperGenerationIsValid)
{
  RoughnessPlaceScanChainTracker tracker({3U, 1.0});
  for (std::int32_t state_id = 1; state_id <= 3; ++state_id) {
    auto stable = scan(state_id, true);
    stable.mapper_generation = 0U;
    tracker.observe(
      stable, RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  }
  for (std::int32_t state_id = 4; state_id <= 5; ++state_id) {
    auto rough = scan(state_id, false);
    rough.mapper_generation = 0U;
    tracker.observe(
      rough,
      state_id == 4 ? RoughnessPlaceTrackerAction::kStarted :
      RoughnessPlaceTrackerAction::kExtended);
  }
  auto post = scan(6, true);
  post.mapper_generation = 0U;
  tracker.observe(post, RoughnessPlaceTrackerAction::kFinalized);
  tracker.finalizeEvent(event(3, 6), {}, post);
  for (std::int32_t state_id = 7; state_id <= 8; ++state_id) {
    auto stable = scan(state_id, true);
    stable.mapper_generation = 0U;
    const auto update = tracker.observe(
      stable, RoughnessPlaceTrackerAction::kStableAnchorUpdated);
    if (state_id == 8) {
      ASSERT_EQ(update.completed_events.size(), 1U);
      EXPECT_TRUE(update.completed_events.front().available);
    }
  }
}

TEST(RoughnessPlaceScanChainTracker, ValidatesConfiguration)
{
  EXPECT_FALSE(validRoughnessPlaceScanChainConfig({1U, 1.0}));
  EXPECT_FALSE(validRoughnessPlaceScanChainConfig({3U, 0.0}));
  EXPECT_FALSE(validRoughnessPlaceScanChainConfig({3U, 1.0, 5U, 0.0}));
  EXPECT_TRUE(validRoughnessPlaceScanChainConfig({3U, 1.0}));
  EXPECT_THROW(
    RoughnessPlaceScanChainTracker({1U, 1.0}),
    std::invalid_argument);
}

}  // namespace
}  // namespace slam_2d_core
