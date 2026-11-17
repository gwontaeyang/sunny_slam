#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "slam_2d_core/karto_loop_closure_observer.hpp"

namespace slam_2d_core
{
namespace
{

constexpr const char * kAcceptedCoarse =
  "COARSE RESPONSE: 0.41 (> 0.2)\n"
  "            var: 0.12,  0.34 (< 20)";
constexpr const char * kRejectedCoarse =
  "COARSE RESPONSE: 0.19 (> 0.2)\n"
  "            var: 0.12,  0.34 (< 20)";

TEST(KartoLoopClosureObserver, ParsesExactKartoMessages)
{
  double response = 0.0;
  double response_threshold = 0.0;
  double covariance_xx = 0.0;
  double covariance_yy = 0.0;
  double covariance_threshold = 0.0;
  ASSERT_TRUE(
    parseKartoLoopClosureCoarseMessage(
      kAcceptedCoarse,
      response,
      response_threshold,
      covariance_xx,
      covariance_yy,
      covariance_threshold));
  EXPECT_DOUBLE_EQ(response, 0.41);
  EXPECT_DOUBLE_EQ(response_threshold, 0.2);
  EXPECT_DOUBLE_EQ(covariance_xx, 0.12);
  EXPECT_DOUBLE_EQ(covariance_yy, 0.34);
  EXPECT_DOUBLE_EQ(covariance_threshold, 20.0);

  ASSERT_TRUE(
    parseKartoLoopClosureFineMessage(
      "FINE RESPONSE: 0.37 (>0.3)\n", response, response_threshold));
  EXPECT_DOUBLE_EQ(response, 0.37);
  EXPECT_DOUBLE_EQ(response_threshold, 0.3);
}

TEST(KartoLoopClosureObserver, RejectsMalformedOrNonfiniteMessages)
{
  double response = 0.0;
  double response_threshold = 0.0;
  double covariance_xx = 0.0;
  double covariance_yy = 0.0;
  double covariance_threshold = 0.0;
  EXPECT_FALSE(
    parseKartoLoopClosureCoarseMessage(
      "COARSE RESPONSE: incomplete",
      response,
      response_threshold,
      covariance_xx,
      covariance_yy,
      covariance_threshold));
  EXPECT_FALSE(
    parseKartoLoopClosureFineMessage(
      "FINE RESPONSE: nan (>0.3)", response, response_threshold));
}

TEST(KartoLoopClosureObserver, ClassifiesCoarseRejectionWithoutExtraCallback)
{
  KartoLoopClosureObserver observer;
  observer.LoopClosureCheck(kRejectedCoarse);
  const auto batch = observer.finish();
  ASSERT_EQ(batch.attempts.size(), 1U);
  const auto & attempt = batch.attempts.front();
  EXPECT_EQ(
    attempt.disposition, KartoLoopClosureDisposition::kCoarseRejected);
  EXPECT_TRUE(attempt.listener_sequence_complete);
  EXPECT_EQ(
    std::string(kartoLoopClosureDispositionName(attempt.disposition)),
    "coarse_rejected");
}

TEST(KartoLoopClosureObserver, ClassifiesFineRejectionSequence)
{
  KartoLoopClosureObserver observer;
  observer.LoopClosureCheck(kAcceptedCoarse);
  observer.LoopClosureCheck("FINE RESPONSE: 0.29 (>0.3)\n");
  observer.LoopClosureCheck("REJECTED!");
  const auto batch = observer.finish();
  ASSERT_EQ(batch.attempts.size(), 1U);
  const auto & attempt = batch.attempts.front();
  EXPECT_EQ(attempt.disposition, KartoLoopClosureDisposition::kFineRejected);
  EXPECT_TRUE(attempt.listener_sequence_complete);
  EXPECT_TRUE(attempt.rejection_callback_seen);
}

TEST(KartoLoopClosureObserver, ClassifiesAcceptedSequence)
{
  KartoLoopClosureObserver observer;
  observer.LoopClosureCheck(kAcceptedCoarse);
  observer.LoopClosureCheck("FINE RESPONSE: 0.31 (>0.3)\n");
  observer.BeginLoopClosure("Closing loop...");
  observer.EndLoopClosure("Loop closed!");
  const auto batch = observer.finish();
  ASSERT_EQ(batch.attempts.size(), 1U);
  const auto & attempt = batch.attempts.front();
  EXPECT_EQ(attempt.disposition, KartoLoopClosureDisposition::kAccepted);
  EXPECT_TRUE(attempt.listener_sequence_complete);
  EXPECT_TRUE(attempt.begin_callback_seen);
  EXPECT_TRUE(attempt.end_callback_seen);
  EXPECT_EQ(batch.accepted_callback_count, 1U);
}

TEST(KartoLoopClosureObserver, DetectsIncompleteAcceptedSequence)
{
  KartoLoopClosureObserver observer;
  observer.LoopClosureCheck(kAcceptedCoarse);
  observer.LoopClosureCheck("FINE RESPONSE: 0.31 (>0.3)\n");
  observer.BeginLoopClosure("Closing loop...");
  const auto batch = observer.finish();
  ASSERT_EQ(batch.attempts.size(), 1U);
  EXPECT_EQ(
    batch.attempts.front().disposition,
    KartoLoopClosureDisposition::kAcceptedIncomplete);
  EXPECT_FALSE(batch.listener_sequence_complete);
  EXPECT_NE(
    batch.failure_reason.find("incomplete_listener_sequence"),
    std::string::npos);
}

}  // namespace
}  // namespace slam_2d_core
