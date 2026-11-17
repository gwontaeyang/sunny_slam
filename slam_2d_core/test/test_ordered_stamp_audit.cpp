#include <cstdint>

#include "gtest/gtest.h"
#include "slam_2d_core/ordered_stamp_audit.hpp"

namespace slam_2d_core
{
namespace
{

TEST(OrderedStampAudit, TracksOrderedSequenceAndFingerprint)
{
  OrderedStampAudit audit;
  audit.record(1);
  audit.record(2);
  audit.record(3);

  const auto snapshot = audit.snapshot();
  EXPECT_EQ(snapshot.count, 3U);
  EXPECT_EQ(snapshot.first_stamp_ns, 1);
  EXPECT_EQ(snapshot.last_stamp_ns, 3);
  EXPECT_EQ(snapshot.max_positive_gap_ns, 1);
  EXPECT_EQ(snapshot.duplicate_count, 0U);
  EXPECT_EQ(snapshot.out_of_order_count, 0U);
  EXPECT_EQ(snapshot.fingerprint_fnv1a64, 15720935049292226309ULL);
}

TEST(OrderedStampAudit, ReportsDuplicatesAndOutOfOrderStamps)
{
  OrderedStampAudit audit;
  audit.record(10);
  audit.record(10);
  audit.record(8);
  audit.record(20);

  const auto snapshot = audit.snapshot();
  EXPECT_EQ(snapshot.count, 4U);
  EXPECT_EQ(snapshot.first_stamp_ns, 10);
  EXPECT_EQ(snapshot.last_stamp_ns, 20);
  EXPECT_EQ(snapshot.max_positive_gap_ns, 12);
  EXPECT_EQ(snapshot.duplicate_count, 1U);
  EXPECT_EQ(snapshot.out_of_order_count, 1U);
}

TEST(OrderedStampAudit, FingerprintDependsOnStampOrder)
{
  OrderedStampAudit first;
  OrderedStampAudit second;
  first.record(100);
  first.record(200);
  second.record(200);
  second.record(100);

  EXPECT_NE(
    first.snapshot().fingerprint_fnv1a64,
    second.snapshot().fingerprint_fnv1a64);
}

}  // namespace
}  // namespace slam_2d_core
