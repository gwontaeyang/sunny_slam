// Copyright 2026 Gole Robotics

#ifndef SLAM_3D_CORE__IMU_SHOCK_POLICY_HPP_
#define SLAM_3D_CORE__IMU_SHOCK_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <string>

namespace gole
{
namespace slam_3d_core
{

enum class ImuShockPhase
{
  kDisabled,
  kStable,
  kShock,
  kRecovering,
};

enum class ImuShockObservation
{
  kUnknown,
  kStable,
  kShock,
};

struct ImuShockPolicyConfig
{
  bool enabled{false};
  double hold_duration_s{0.5};
  int recovery_stable_scans{3};
};

struct ImuShockPolicyState
{
  ImuShockPhase phase{ImuShockPhase::kDisabled};
  double last_update_stamp_s{0.0};
  double last_event_stamp_s{0.0};
  int recovery_stable_scan_count{0};
  bool have_last_update_stamp{false};
  bool have_last_event_stamp{false};
};

struct ImuShockPolicyDecision
{
  ImuShockPhase phase{ImuShockPhase::kDisabled};
  ImuShockObservation observation{ImuShockObservation::kUnknown};
  bool active{false};
  bool time_reset{false};
  bool just_recovered{false};
  int recovery_stable_scan_count{0};
  double time_since_event_s{0.0};
};

inline const char * toString(const ImuShockPhase phase)
{
  switch (phase) {
    case ImuShockPhase::kDisabled:
      return "disabled";
    case ImuShockPhase::kStable:
      return "stable";
    case ImuShockPhase::kShock:
      return "shock";
    case ImuShockPhase::kRecovering:
      return "recovering";
  }
  return "unknown";
}

inline const char * toString(const ImuShockObservation observation)
{
  switch (observation) {
    case ImuShockObservation::kUnknown:
      return "unknown";
    case ImuShockObservation::kStable:
      return "stable";
    case ImuShockObservation::kShock:
      return "shock";
  }
  return "unknown";
}

inline ImuShockPolicyDecision updateImuShockPolicy(
  const ImuShockPolicyConfig & config,
  const ImuShockObservation observation,
  const double stamp_s,
  ImuShockPolicyState & state)
{
  ImuShockPolicyDecision decision;
  decision.observation = observation;
  if (!config.enabled) {
    state = ImuShockPolicyState{};
    return decision;
  }

  if (!std::isfinite(stamp_s)) {
    state = ImuShockPolicyState{};
    state.phase = ImuShockPhase::kShock;
    decision.phase = state.phase;
    decision.active = true;
    decision.time_reset = true;
    return decision;
  }

  const bool have_previous_stamp = state.have_last_update_stamp;
  const bool timestamp_rewind =
    have_previous_stamp && stamp_s < state.last_update_stamp_s;
  const bool duplicate_timestamp =
    have_previous_stamp && stamp_s == state.last_update_stamp_s;
  if (timestamp_rewind) {
    state = ImuShockPolicyState{};
    state.phase = ImuShockPhase::kShock;
    state.last_update_stamp_s = stamp_s;
    state.last_event_stamp_s = stamp_s;
    state.have_last_update_stamp = true;
    state.have_last_event_stamp = true;
    decision.phase = state.phase;
    decision.active = true;
    decision.time_reset = true;
    return decision;
  }
  state.last_update_stamp_s = stamp_s;
  state.have_last_update_stamp = true;

  const bool shock_observed = observation == ImuShockObservation::kShock;
  const bool stable_observed = observation == ImuShockObservation::kStable;
  if (shock_observed) {
    state.phase = ImuShockPhase::kShock;
    state.last_event_stamp_s = stamp_s;
    state.have_last_event_stamp = true;
    state.recovery_stable_scan_count = 0;
  } else if (state.phase == ImuShockPhase::kDisabled) {
    state.phase = ImuShockPhase::kStable;
  }

  const double hold_duration_s = std::max(0.0, config.hold_duration_s);
  const int required_stable_scans = std::max(1, config.recovery_stable_scans);
  if (state.phase == ImuShockPhase::kShock && stable_observed && !duplicate_timestamp) {
    if (!state.have_last_event_stamp) {
      // A non-finite timestamp cannot define the beginning of the hold period.
      // Start the hold clock at the first subsequent finite observation.
      state.last_event_stamp_s = stamp_s;
      state.have_last_event_stamp = true;
    } else if (stamp_s - state.last_event_stamp_s >= hold_duration_s) {
      state.phase = ImuShockPhase::kRecovering;
      state.recovery_stable_scan_count = 1;
    }
  } else if (
    state.phase == ImuShockPhase::kRecovering && stable_observed && !duplicate_timestamp)
  {
    ++state.recovery_stable_scan_count;
  } else if (state.phase == ImuShockPhase::kRecovering) {
    // Recovery means consecutive, fresh stable observations. Missing IMU
    // coverage and duplicate timestamps keep the policy fail-closed.
    state.phase = ImuShockPhase::kShock;
    state.recovery_stable_scan_count = 0;
    state.last_event_stamp_s = stamp_s;
    state.have_last_event_stamp = true;
  }

  if (state.phase == ImuShockPhase::kRecovering &&
    state.recovery_stable_scan_count >= required_stable_scans)
  {
    state.phase = ImuShockPhase::kStable;
    state.recovery_stable_scan_count = required_stable_scans;
    decision.just_recovered = true;
  }

  decision.phase = state.phase;
  decision.active =
    state.phase == ImuShockPhase::kShock || state.phase == ImuShockPhase::kRecovering;
  decision.recovery_stable_scan_count = state.recovery_stable_scan_count;
  decision.time_since_event_s = state.have_last_event_stamp ?
    std::max(0.0, stamp_s - state.last_event_stamp_s) : 0.0;
  return decision;
}

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__IMU_SHOCK_POLICY_HPP_
