#ifndef SLAM_2D_CORE__ROUGHNESS_PLACE_SCAN_CHAIN_HPP_
#define SLAM_2D_CORE__ROUGHNESS_PLACE_SCAN_CHAIN_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "slam_2d_core/roughness_place_recognition.hpp"

namespace slam_2d_core
{

struct RoughnessPlaceScanChainConfig
{
  std::size_t scan_count{5U};
  double maximum_anchor_distance_m{1.0};
  std::size_t maximum_anchor_state_gap{5U};
  double maximum_anchor_time_gap_s{5.0};
};

bool validRoughnessPlaceScanChainConfig(
  const RoughnessPlaceScanChainConfig & config) noexcept;

struct RoughnessPlaceStableScan
{
  std::uint64_t mapper_generation{0U};
  std::string sensor_name;
  std::int64_t stamp_ns{0};
  std::int32_t state_id{-1};
  std::int32_t unique_id{-1};
  double odom_x_m{0.0};
  double odom_y_m{0.0};
  double odom_yaw_rad{0.0};
  bool quality_valid{false};
  bool strict_stable{false};
};

struct RoughnessPlaceScanChainEvent
{
  bool available{false};
  std::string failure_reason{"not_evaluated"};
  RoughnessPlaceEvent event;
  std::vector<RoughnessPlaceCandidate> candidates;
  std::vector<RoughnessPlaceStableScan> pre_chain;
  std::vector<RoughnessPlaceStableScan> post_chain;
  std::size_t post_reacquisition_reset_count{0U};
};

struct RoughnessPlaceScanChainUpdate
{
  std::vector<RoughnessPlaceScanChainEvent> completed_events;
};

class RoughnessPlaceScanChainTracker
{
public:
  explicit RoughnessPlaceScanChainTracker(
    RoughnessPlaceScanChainConfig config = {});

  void configure(const RoughnessPlaceScanChainConfig & config);
  void reset();
  RoughnessPlaceScanChainUpdate observe(
    const RoughnessPlaceStableScan & scan,
    RoughnessPlaceTrackerAction tracker_action);
  RoughnessPlaceScanChainUpdate finalizeEvent(
    const RoughnessPlaceEvent & event,
    std::vector<RoughnessPlaceCandidate> candidates,
    const RoughnessPlaceStableScan & post_anchor);

private:
  void appendStableHistory(const RoughnessPlaceStableScan & scan);
  bool chainWithinDistance(
    const std::vector<RoughnessPlaceStableScan> & chain,
    bool anchor_is_last) const noexcept;
  RoughnessPlaceScanChainEvent complete(
    RoughnessPlaceScanChainEvent event,
    const char * failure_reason) const;
  const char * postReacquisitionFailure(
    const RoughnessPlaceScanChainEvent & pending,
    const RoughnessPlaceStableScan & scan) const noexcept;

  RoughnessPlaceScanChainConfig config_;
  std::deque<RoughnessPlaceStableScan> stable_history_;
  std::vector<RoughnessPlaceStableScan> last_complete_stable_chain_;
  std::vector<RoughnessPlaceStableScan> active_pre_chain_;
  std::vector<RoughnessPlaceScanChainEvent> pending_events_;
  std::int64_t last_stamp_ns_{0};
  std::uint64_t mapper_generation_{0U};
  bool mapper_generation_initialized_{false};
};

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__ROUGHNESS_PLACE_SCAN_CHAIN_HPP_
