#include <gtest/gtest.h>

#include <vector>

#include "slam_2d_core/karto_near_chain_observer.hpp"

namespace slam_2d_core
{
namespace
{

NearChainEdgeEndpoint edge(
  const std::int32_t source,
  const std::int32_t target,
  const bool link_info = true)
{
  return NearChainEdgeEndpoint{source, target, link_info};
}

TEST(KartoNearChainObserver, AccountsForPrelinkedAndCandidateEdges)
{
  const auto result = evaluateNearChainEdgeAccounting(
    100,
    {10, 20},
    {99},
    {},
    {edge(99, 100), edge(10, 100), edge(20, 100)});
  EXPECT_TRUE(result.complete);
  EXPECT_TRUE(result.candidate_endpoints_unique);
  EXPECT_EQ(result.classified_edge_count, 3U);
  EXPECT_EQ(result.candidate_edge_count, 2U);
  EXPECT_EQ(result.unexpected_edge_count, 0U);
}

TEST(KartoNearChainObserver, FailsClosedForUnexpectedCurrentEdge)
{
  const auto result = evaluateNearChainEdgeAccounting(
    100,
    {10},
    {99},
    {},
    {edge(99, 100), edge(42, 100)});
  EXPECT_FALSE(result.complete);
  EXPECT_EQ(result.classified_edge_count, 1U);
  EXPECT_EQ(result.unexpected_edge_count, 1U);
}

TEST(KartoNearChainObserver, FailsClosedForDuplicateCandidateEndpoint)
{
  const auto result = evaluateNearChainEdgeAccounting(
    100,
    {10, 10},
    {99},
    {},
    {edge(99, 100), edge(10, 100)});
  EXPECT_FALSE(result.complete);
  EXPECT_FALSE(result.candidate_endpoints_unique);
}

TEST(KartoNearChainObserver, FailsClosedForMalformedOrUnrelatedEdge)
{
  const auto result = evaluateNearChainEdgeAccounting(
    100,
    {10},
    {99},
    {},
    {edge(10, 100, false), edge(1, 2)});

  EXPECT_FALSE(result.complete);
  EXPECT_EQ(result.unexpected_edge_count, 2U);
}

TEST(KartoNearChainObserver, AcceptsExternallyClassifiedLoopEdge)
{
  const auto result = evaluateNearChainEdgeAccounting(
    100,
    {10},
    {99},
    {42},
    {edge(99, 100), edge(10, 100), edge(42, 100)});
  EXPECT_TRUE(result.complete);
  EXPECT_EQ(result.classified_edge_count, 3U);
  EXPECT_EQ(result.candidate_edge_count, 1U);
  EXPECT_EQ(result.unexpected_edge_count, 0U);
}

}  // namespace
}  // namespace slam_2d_core
