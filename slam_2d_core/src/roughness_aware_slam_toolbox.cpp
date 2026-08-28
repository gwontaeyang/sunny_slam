#include "slam_2d_core/roughness_aware_slam_toolbox.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "rmw/qos_profiles.h"
#include "tf2/exceptions.h"

namespace slam_2d_core
{
namespace
{

constexpr double kGravityMps2 = 9.80665;
constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
constexpr double kParameterComparisonTolerance = 1.0e-12;

class ScopedLinkMatchMinimumResponse
{
public:
  ScopedLinkMatchMinimumResponse(
    karto::Mapper * mapper,
    const bool enable_override,
    const double requested)
  : mapper_(mapper),
    baseline_(mapper ? mapper->getParamLinkMatchMinimumResponseFine() :
      std::numeric_limits<double>::quiet_NaN()),
    applied_(baseline_)
  {
    if (
      !mapper_ || !enable_override ||
      !std::isfinite(baseline_) || !std::isfinite(requested))
    {
      return;
    }
    mapper_->setParamLinkMatchMinimumResponseFine(requested);
    applied_ = mapper_->getParamLinkMatchMinimumResponseFine();
    overridden_ =
      std::isfinite(applied_) &&
      std::fabs(applied_ - baseline_) > kParameterComparisonTolerance;
    restore_pending_ = true;
  }

  ~ScopedLinkMatchMinimumResponse()
  {
    (void)restore();
  }

  ScopedLinkMatchMinimumResponse(const ScopedLinkMatchMinimumResponse &) = delete;
  ScopedLinkMatchMinimumResponse & operator=(
    const ScopedLinkMatchMinimumResponse &) = delete;

  bool restore() noexcept
  {
    if (!restore_pending_) {
      return restore_succeeded_;
    }
    restore_pending_ = false;
    try {
      mapper_->setParamLinkMatchMinimumResponseFine(baseline_);
      restored_ = mapper_->getParamLinkMatchMinimumResponseFine();
      restore_succeeded_ =
        std::isfinite(restored_) &&
        std::fabs(restored_ - baseline_) <= kParameterComparisonTolerance;
    } catch (...) {
      restore_succeeded_ = false;
    }
    return restore_succeeded_;
  }

  double baseline() const {return baseline_;}
  double applied() const {return applied_;}
  double restored() const {return restored_;}
  bool overridden() const {return overridden_;}

private:
  karto::Mapper * mapper_{nullptr};
  double baseline_{std::numeric_limits<double>::quiet_NaN()};
  double applied_{std::numeric_limits<double>::quiet_NaN()};
  double restored_{std::numeric_limits<double>::quiet_NaN()};
  bool overridden_{false};
  bool restore_pending_{false};
  bool restore_succeeded_{true};
};

class ScopedLoopClosingSuppression
{
public:
  ScopedLoopClosingSuppression(
    karto::Mapper * mapper,
    const bool suppress)
  : mapper_(mapper),
    baseline_(mapper ? mapper->getParamDoLoopClosing() : false),
    applied_(baseline_),
    restored_(baseline_)
  {
    if (!mapper_ || !suppress || !baseline_) {
      return;
    }
    mapper_->setParamDoLoopClosing(false);
    applied_ = mapper_->getParamDoLoopClosing();
    suppressed_ = !applied_;
    restore_pending_ = true;
  }

  ~ScopedLoopClosingSuppression()
  {
    (void)restore();
  }

  ScopedLoopClosingSuppression(const ScopedLoopClosingSuppression &) = delete;
  ScopedLoopClosingSuppression & operator=(
    const ScopedLoopClosingSuppression &) = delete;

  bool restore() noexcept
  {
    if (!restore_pending_) {
      return restore_succeeded_;
    }
    restore_pending_ = false;
    try {
      mapper_->setParamDoLoopClosing(baseline_);
      restored_ = mapper_->getParamDoLoopClosing();
      restore_succeeded_ = restored_ == baseline_;
    } catch (...) {
      restore_succeeded_ = false;
    }
    return restore_succeeded_;
  }

  bool baseline() const {return baseline_;}
  bool applied() const {return applied_;}
  bool restored() const {return restored_;}
  bool suppressed() const {return suppressed_;}

private:
  karto::Mapper * mapper_{nullptr};
  bool baseline_{false};
  bool applied_{false};
  bool restored_{false};
  bool suppressed_{false};
  bool restore_pending_{false};
  bool restore_succeeded_{true};
};

bool finiteVector(const geometry_msgs::msg::Vector3 & vector)
{
  return
    std::isfinite(vector.x) &&
    std::isfinite(vector.y) &&
    std::isfinite(vector.z);
}

double normalizeAngleValue(double angle_rad) noexcept
{
  while (angle_rad > 3.14159265358979323846) {
    angle_rad -= 2.0 * 3.14159265358979323846;
  }
  while (angle_rad < -3.14159265358979323846) {
    angle_rad += 2.0 * 3.14159265358979323846;
  }
  return angle_rad;
}

karto::Pose2 unavailablePose() noexcept
{
  const double value = std::numeric_limits<double>::quiet_NaN();
  return karto::Pose2(value, value, value);
}

void setMatrixUnavailable(karto::Matrix3 & matrix) noexcept
{
  const double value = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      matrix(row, column) = value;
    }
  }
}

bool finitePose(const karto::Pose2 & pose) noexcept
{
  return
    std::isfinite(pose.GetX()) &&
    std::isfinite(pose.GetY()) &&
    std::isfinite(pose.GetHeading());
}

RoughnessPlacePose2D roughnessPlacePose(
  const karto::Pose2 & pose,
  const double yaw_offset_rad = 0.0) noexcept
{
  return {
    pose.GetX(),
    pose.GetY(),
    normalizeAngleValue(pose.GetHeading() + yaw_offset_rad),
  };
}

bool finiteMatrix(const karto::Matrix3 & matrix) noexcept
{
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      if (!std::isfinite(matrix(row, column))) {
        return false;
      }
    }
  }
  return true;
}

karto::Pose2 relativePose(
  const karto::Pose2 & from,
  const karto::Pose2 & to)
{
  return karto::Transform(from, karto::Pose2()).TransformPose(to);
}

karto::Pose2 composePose(
  const karto::Pose2 & first,
  const karto::Pose2 & second) noexcept
{
  const double cosine = std::cos(first.GetHeading());
  const double sine = std::sin(first.GetHeading());
  return karto::Pose2(
    first.GetX() + cosine * second.GetX() - sine * second.GetY(),
    first.GetY() + sine * second.GetX() + cosine * second.GetY(),
    normalizeAngleValue(first.GetHeading() + second.GetHeading()));
}

karto::Pose2 inversePose(const karto::Pose2 & pose) noexcept
{
  const double cosine = std::cos(pose.GetHeading());
  const double sine = std::sin(pose.GetHeading());
  return karto::Pose2(
    -cosine * pose.GetX() - sine * pose.GetY(),
    sine * pose.GetX() - cosine * pose.GetY(),
    normalizeAngleValue(-pose.GetHeading()));
}

double poseTranslationDifference(
  const karto::Pose2 & first,
  const karto::Pose2 & second) noexcept
{
  return std::hypot(
    first.GetX() - second.GetX(),
    first.GetY() - second.GetY());
}

struct RoughnessPlaceDirectionalScanMatch
{
  bool available{false};
  std::string failure_reason{"not_evaluated"};
  double coarse_response{std::numeric_limits<double>::quiet_NaN()};
  double fine_response{std::numeric_limits<double>::quiet_NaN()};
  karto::Pose2 proposal_robot_pose{unavailablePose()};
  karto::Pose2 matched_robot_pose{unavailablePose()};
  karto::Matrix3 coarse_covariance;
  karto::Matrix3 fine_covariance;
  double proposal_disagreement_translation_m{
    std::numeric_limits<double>::quiet_NaN()};
  double proposal_disagreement_yaw_rad{
    std::numeric_limits<double>::quiet_NaN()};

  RoughnessPlaceDirectionalScanMatch()
  {
    setMatrixUnavailable(coarse_covariance);
    setMatrixUnavailable(fine_covariance);
  }
};

struct RoughnessPlaceAnchorScanMatch
{
  bool available{false};
  std::string failure_reason{"not_evaluated"};
  RoughnessPlaceDirectionalScanMatch forward;
  RoughnessPlaceDirectionalScanMatch reverse;
  karto::Pose2 odometric_relative_transform{unavailablePose()};
  karto::Pose2 forward_relative_transform{unavailablePose()};
  karto::Pose2 reverse_inverted_relative_transform{unavailablePose()};
  karto::Pose2 forward_branch_alignment{unavailablePose()};
  karto::Pose2 reverse_branch_alignment{unavailablePose()};
  double measurement_vs_odometry_translation_m{
    std::numeric_limits<double>::quiet_NaN()};
  double measurement_vs_odometry_yaw_rad{
    std::numeric_limits<double>::quiet_NaN()};
  double bidirectional_translation_error_m{
    std::numeric_limits<double>::quiet_NaN()};
  double bidirectional_yaw_error_rad{
    std::numeric_limits<double>::quiet_NaN()};
  double alignment_translation_error_m{
    std::numeric_limits<double>::quiet_NaN()};
  double alignment_yaw_error_rad{
    std::numeric_limits<double>::quiet_NaN()};
};

RoughnessPlaceDirectionalScanMatch observeDirectionalPlaceScanMatch(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * moving_scan,
  const karto::LocalizedRangeScanVector & base_scans,
  const karto::Pose2 & proposal_robot_pose) noexcept
{
  RoughnessPlaceDirectionalScanMatch result;
  result.proposal_robot_pose = proposal_robot_pose;
  if (
    !mapper || !moving_scan || base_scans.empty() ||
    std::any_of(
      base_scans.begin(), base_scans.end(),
      [](const karto::LocalizedRangeScan * scan) {return scan == nullptr;}))
  {
    result.failure_reason = "missing_mapper_or_scan";
    return result;
  }
  if (!finitePose(proposal_robot_pose)) {
    result.failure_reason = "invalid_proposal_pose";
    return result;
  }
  auto * coarse_matcher = mapper->GetLoopScanMatcher();
  auto * fine_matcher = mapper->GetSequentialScanMatcher();
  if (!coarse_matcher || !fine_matcher) {
    result.failure_reason = "missing_scan_matcher";
    return result;
  }

  try {
    karto::LocalizedRangeScan temporary_scan(
      moving_scan->GetSensorName(),
      moving_scan->GetRangeReadingsVector());
    temporary_scan.SetUniqueId(moving_scan->GetUniqueId());
    temporary_scan.SetStateId(moving_scan->GetStateId());
    temporary_scan.SetTime(moving_scan->GetTime());
    temporary_scan.SetOdometricPose(moving_scan->GetOdometricPose());
    temporary_scan.SetCorrectedPose(proposal_robot_pose);

    karto::Pose2 matched_sensor_pose;
    result.coarse_response = coarse_matcher->MatchScan(
      &temporary_scan,
      base_scans,
      matched_sensor_pose,
      result.coarse_covariance,
      false,
      false);
    if (
      !std::isfinite(result.coarse_response) ||
      !finitePose(matched_sensor_pose) ||
      !finiteMatrix(result.coarse_covariance))
    {
      result.failure_reason = "non_finite_coarse_match";
      return result;
    }

    temporary_scan.SetSensorPose(matched_sensor_pose);
    result.fine_response = fine_matcher->MatchScan(
      &temporary_scan,
      base_scans,
      matched_sensor_pose,
      result.fine_covariance,
      false,
      true);
    result.matched_robot_pose = temporary_scan.GetCorrectedAt(
      matched_sensor_pose);
    if (
      !std::isfinite(result.fine_response) ||
      !finitePose(result.matched_robot_pose) ||
      !finiteMatrix(result.fine_covariance))
    {
      result.failure_reason = "non_finite_fine_match";
      return result;
    }

    result.proposal_disagreement_translation_m = poseTranslationDifference(
      result.matched_robot_pose, proposal_robot_pose);
    result.proposal_disagreement_yaw_rad = std::fabs(
      normalizeAngleValue(
        result.matched_robot_pose.GetHeading() -
        proposal_robot_pose.GetHeading()));
    result.available = true;
    result.failure_reason = "none";
  } catch (const karto::Exception & ex) {
    result.failure_reason = "karto_exception:" + ex.GetErrorMessage();
  } catch (const std::exception & ex) {
    result.failure_reason = "exception:" + std::string(ex.what());
  } catch (...) {
    result.failure_reason = "unknown_exception";
  }
  return result;
}

RoughnessPlaceAnchorScanMatch observeRoughnessPlaceAnchorScanMatch(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * reference_scan,
  karto::LocalizedRangeScan * query_scan,
  const karto::LocalizedRangeScanVector & reference_base_scans,
  const karto::LocalizedRangeScanVector & query_base_scans) noexcept
{
  RoughnessPlaceAnchorScanMatch result;
  if (
    !mapper || !reference_scan || !query_scan ||
    reference_base_scans.empty() || query_base_scans.empty())
  {
    result.failure_reason = "missing_mapper_or_anchor";
    return result;
  }
  const karto::Pose2 & reference_odom = reference_scan->GetOdometricPose();
  const karto::Pose2 & query_odom = query_scan->GetOdometricPose();
  const karto::Pose2 & reference_corrected = reference_scan->GetCorrectedPose();
  const karto::Pose2 & query_corrected = query_scan->GetCorrectedPose();
  if (
    !finitePose(reference_odom) || !finitePose(query_odom) ||
    !finitePose(reference_corrected) || !finitePose(query_corrected))
  {
    result.failure_reason = "invalid_anchor_pose";
    return result;
  }

  result.odometric_relative_transform = relativePose(
    reference_odom, query_odom);
  const karto::Pose2 forward_proposal = composePose(
    reference_corrected, result.odometric_relative_transform);
  result.forward = observeDirectionalPlaceScanMatch(
    mapper, query_scan, reference_base_scans, forward_proposal);

  const karto::Pose2 reverse_odometric_relative = relativePose(
    query_odom, reference_odom);
  const karto::Pose2 reverse_proposal = composePose(
    query_corrected, reverse_odometric_relative);
  result.reverse = observeDirectionalPlaceScanMatch(
    mapper, reference_scan, query_base_scans, reverse_proposal);

  if (result.forward.available) {
    result.forward_relative_transform = relativePose(
      reference_corrected, result.forward.matched_robot_pose);
    result.forward_branch_alignment = composePose(
      result.forward.matched_robot_pose, inversePose(query_corrected));
    result.measurement_vs_odometry_translation_m = poseTranslationDifference(
      result.forward_relative_transform,
      result.odometric_relative_transform);
    result.measurement_vs_odometry_yaw_rad = std::fabs(
      normalizeAngleValue(
        result.forward_relative_transform.GetHeading() -
        result.odometric_relative_transform.GetHeading()));
  }
  if (result.reverse.available) {
    const karto::Pose2 reverse_relative_transform = relativePose(
      query_corrected, result.reverse.matched_robot_pose);
    result.reverse_inverted_relative_transform = inversePose(
      reverse_relative_transform);
    result.reverse_branch_alignment = composePose(
      reference_corrected, inversePose(result.reverse.matched_robot_pose));
  }
  if (result.forward.available && result.reverse.available) {
    result.bidirectional_translation_error_m = poseTranslationDifference(
      result.forward_relative_transform,
      result.reverse_inverted_relative_transform);
    result.bidirectional_yaw_error_rad = std::fabs(
      normalizeAngleValue(
        result.forward_relative_transform.GetHeading() -
        result.reverse_inverted_relative_transform.GetHeading()));
    result.alignment_translation_error_m = poseTranslationDifference(
      result.forward_branch_alignment,
      result.reverse_branch_alignment);
    result.alignment_yaw_error_rad = std::fabs(
      normalizeAngleValue(
        result.forward_branch_alignment.GetHeading() -
        result.reverse_branch_alignment.GetHeading()));
    result.available = true;
    result.failure_reason = "none";
  } else {
    result.failure_reason =
      "forward=" + result.forward.failure_reason +
      ",reverse=" + result.reverse.failure_reason;
  }
  return result;
}

RoughnessPlaceAnchorScanMatch observeRoughnessPlaceAnchorScanMatch(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * reference_scan,
  karto::LocalizedRangeScan * query_scan) noexcept
{
  return observeRoughnessPlaceAnchorScanMatch(
    mapper,
    reference_scan,
    query_scan,
    karto::LocalizedRangeScanVector{reference_scan},
    karto::LocalizedRangeScanVector{query_scan});
}

karto::LocalizedRangeScan * findGraphScan(
  karto::Mapper * mapper,
  const karto::Name & sensor_name,
  const std::int32_t state_id) noexcept
{
  if (!mapper || !mapper->GetMapperSensorManager() || state_id < 0) {
    return nullptr;
  }
  try {
    return mapper->GetMapperSensorManager()->GetScan(sensor_name, state_id);
  } catch (...) {
    return nullptr;
  }
}

struct FrozenRoughnessPlaceScanChain
{
  bool available{false};
  std::string failure_reason{"not_evaluated"};
  std::vector<std::unique_ptr<karto::LocalizedRangeScan>> storage;
  karto::LocalizedRangeScanVector scans;
  std::int32_t first_state_id{-1};
  std::int32_t last_state_id{-1};
  double maximum_anchor_distance_m{
    std::numeric_limits<double>::quiet_NaN()};
};

FrozenRoughnessPlaceScanChain freezeRoughnessPlaceScanChain(
  karto::Mapper * mapper,
  const karto::Name & sensor_name,
  const std::vector<RoughnessPlaceStableScan> & source_chain,
  const std::int32_t anchor_state_id,
  const std::uint64_t expected_mapper_generation) noexcept
{
  FrozenRoughnessPlaceScanChain result;
  if (!mapper || source_chain.empty() || anchor_state_id < 0) {
    result.failure_reason = "missing_mapper_or_chain";
    return result;
  }
  karto::LocalizedRangeScan * anchor_scan = findGraphScan(
    mapper, sensor_name, anchor_state_id);
  if (
    !anchor_scan || !finitePose(anchor_scan->GetOdometricPose()) ||
    !finitePose(anchor_scan->GetCorrectedPose()))
  {
    result.failure_reason = "missing_or_invalid_anchor";
    return result;
  }
  const karto::Pose2 anchor_odom = anchor_scan->GetOdometricPose();
  const karto::Pose2 anchor_corrected = anchor_scan->GetCorrectedPose();
  result.storage.reserve(source_chain.size());
  result.scans.reserve(source_chain.size());
  result.maximum_anchor_distance_m = 0.0;
  for (const RoughnessPlaceStableScan & key : source_chain) {
    if (
      key.sensor_name != sensor_name.ToString() ||
      key.mapper_generation != expected_mapper_generation)
    {
      result.failure_reason = "chain_identity_mismatch";
      return result;
    }
    karto::LocalizedRangeScan * source = findGraphScan(
      mapper, sensor_name, key.state_id);
    if (
      !source || source->GetUniqueId() != key.unique_id ||
      !finitePose(source->GetOdometricPose()))
    {
      result.failure_reason = "chain_scan_lookup_mismatch";
      return result;
    }
    std::unique_ptr<karto::LocalizedRangeScan> clone(
      new karto::LocalizedRangeScan(
        source->GetSensorName(), source->GetRangeReadingsVector()));
    clone->SetUniqueId(source->GetUniqueId());
    clone->SetStateId(source->GetStateId());
    clone->SetTime(source->GetTime());
    clone->SetOdometricPose(source->GetOdometricPose());
    const karto::Pose2 odometric_relative = relativePose(
      anchor_odom, source->GetOdometricPose());
    clone->SetCorrectedPose(composePose(anchor_corrected, odometric_relative));
    result.maximum_anchor_distance_m = std::max(
      result.maximum_anchor_distance_m,
      poseTranslationDifference(source->GetOdometricPose(), anchor_odom));
    result.scans.push_back(clone.get());
    result.storage.push_back(std::move(clone));
  }
  result.first_state_id = source_chain.front().state_id;
  result.last_state_id = source_chain.back().state_id;
  const std::size_t anchor_count = static_cast<std::size_t>(std::count_if(
      source_chain.begin(), source_chain.end(),
      [anchor_state_id](const RoughnessPlaceStableScan & key) {
        return key.state_id == anchor_state_id;
      }));
  if (anchor_count != 1U) {
    result.failure_reason = "anchor_not_unique_in_chain";
    return result;
  }
  result.available = true;
  result.failure_reason = "none";
  return result;
}

const std::vector<RoughnessPlaceStableScan> * roughnessPlaceChainForAnchor(
  const RoughnessPlaceScanChainEvent & context,
  const std::int32_t anchor_state_id) noexcept
{
  if (
    !context.pre_chain.empty() &&
    anchor_state_id == context.pre_chain.back().state_id)
  {
    return &context.pre_chain;
  }
  if (
    !context.post_chain.empty() &&
    anchor_state_id == context.post_chain.front().state_id)
  {
    return &context.post_chain;
  }
  return nullptr;
}

RoughnessPlaceAnchorPairing roughnessPlaceScanChainPairing(
  const RoughnessPlaceScanChainEvent & reference,
  const RoughnessPlaceScanChainEvent & query,
  const RoughnessPlaceMatchDirection direction) noexcept
{
  RoughnessPlaceAnchorPairing pairing;
  if (
    reference.pre_chain.empty() || reference.post_chain.empty() ||
    query.pre_chain.empty() || query.post_chain.empty())
  {
    return pairing;
  }
  pairing.reference_pre_state_id = reference.pre_chain.back().state_id;
  pairing.reference_post_state_id = reference.post_chain.front().state_id;
  if (direction == RoughnessPlaceMatchDirection::kReverse) {
    pairing.query_for_reference_pre_state_id = query.post_chain.front().state_id;
    pairing.query_for_reference_post_state_id = query.pre_chain.back().state_id;
  } else {
    pairing.query_for_reference_pre_state_id = query.pre_chain.back().state_id;
    pairing.query_for_reference_post_state_id = query.post_chain.front().state_id;
  }
  pairing.valid =
    pairing.reference_pre_state_id >= 0 &&
    pairing.reference_post_state_id >= 0 &&
    pairing.query_for_reference_pre_state_id >= 0 &&
    pairing.query_for_reference_post_state_id >= 0;
  return pairing;
}

struct RoughnessPlaceScanChainAnchorEvaluation
{
  bool available{false};
  std::string failure_reason{"not_evaluated"};
  RoughnessPlaceAnchorScanMatch match;
  std::size_t reference_base_scan_count{0U};
  std::size_t query_base_scan_count{0U};
  std::int32_t reference_first_state_id{-1};
  std::int32_t reference_last_state_id{-1};
  std::int32_t query_first_state_id{-1};
  std::int32_t query_last_state_id{-1};
  double reference_maximum_anchor_distance_m{
    std::numeric_limits<double>::quiet_NaN()};
  double query_maximum_anchor_distance_m{
    std::numeric_limits<double>::quiet_NaN()};
};

RoughnessPlaceScanChainAnchorEvaluation observeRoughnessPlaceScanChainAnchor(
  karto::Mapper * mapper,
  const karto::Name & sensor_name,
  karto::LocalizedRangeScan * reference_scan,
  karto::LocalizedRangeScan * query_scan,
  const std::vector<RoughnessPlaceStableScan> * reference_chain,
  const std::vector<RoughnessPlaceStableScan> * query_chain,
  const std::uint64_t expected_mapper_generation) noexcept
{
  RoughnessPlaceScanChainAnchorEvaluation result;
  if (
    !mapper || !reference_scan || !query_scan ||
    !reference_chain || !query_chain)
  {
    result.failure_reason = "missing_mapper_anchor_or_chain";
    return result;
  }
  FrozenRoughnessPlaceScanChain frozen_reference =
    freezeRoughnessPlaceScanChain(
    mapper,
    sensor_name,
    *reference_chain,
    reference_scan->GetStateId(),
    expected_mapper_generation);
  FrozenRoughnessPlaceScanChain frozen_query = freezeRoughnessPlaceScanChain(
    mapper,
    sensor_name,
    *query_chain,
    query_scan->GetStateId(),
    expected_mapper_generation);
  result.reference_base_scan_count = frozen_reference.scans.size();
  result.query_base_scan_count = frozen_query.scans.size();
  result.reference_first_state_id = frozen_reference.first_state_id;
  result.reference_last_state_id = frozen_reference.last_state_id;
  result.query_first_state_id = frozen_query.first_state_id;
  result.query_last_state_id = frozen_query.last_state_id;
  result.reference_maximum_anchor_distance_m =
    frozen_reference.maximum_anchor_distance_m;
  result.query_maximum_anchor_distance_m =
    frozen_query.maximum_anchor_distance_m;
  if (!frozen_reference.available || !frozen_query.available) {
    result.failure_reason =
      "reference=" + frozen_reference.failure_reason +
      ",query=" + frozen_query.failure_reason;
    return result;
  }
  result.match = observeRoughnessPlaceAnchorScanMatch(
    mapper,
    reference_scan,
    query_scan,
    frozen_reference.scans,
    frozen_query.scans);
  result.available = result.match.available;
  result.failure_reason = result.available ? "none" : result.match.failure_reason;
  return result;
}

double covarianceXyMaximumEigenvalue(const karto::Matrix3 & covariance) noexcept
{
  if (!finiteMatrix(covariance)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double difference = covariance(0, 0) - covariance(1, 1);
  const double discriminant = std::sqrt(
    std::max(
      0.0,
      difference * difference +
      4.0 * covariance(0, 1) * covariance(0, 1)));
  return 0.5 * (covariance(0, 0) + covariance(1, 1) + discriminant);
}

RoughnessPlaceDirectionalMatchEvidence placeDirectionalEvidence(
  const RoughnessPlaceDirectionalScanMatch & match) noexcept
{
  RoughnessPlaceDirectionalMatchEvidence evidence;
  evidence.available = match.available;
  evidence.fine_response = match.fine_response;
  evidence.fine_xy_covariance_max_eigenvalue_m2 =
    covarianceXyMaximumEigenvalue(match.fine_covariance);
  evidence.fine_yaw_covariance_rad2 = match.fine_covariance(2, 2);
  evidence.proposal_disagreement_translation_m =
    match.proposal_disagreement_translation_m;
  evidence.proposal_disagreement_yaw_rad =
    match.proposal_disagreement_yaw_rad;
  return evidence;
}

RoughnessPlaceAnchorMatchEvidence placeAnchorEvidence(
  const RoughnessPlaceAnchorScanMatch & match) noexcept
{
  RoughnessPlaceAnchorMatchEvidence evidence;
  evidence.available = match.available;
  evidence.forward = placeDirectionalEvidence(match.forward);
  evidence.reverse = placeDirectionalEvidence(match.reverse);
  evidence.bidirectional_translation_error_m =
    match.bidirectional_translation_error_m;
  evidence.bidirectional_yaw_error_rad = match.bidirectional_yaw_error_rad;
  return evidence;
}

RoughnessPlacePairMatchEvidence placePairEvidence(
  const RoughnessPlaceAnchorScanMatch & pre_anchor,
  const RoughnessPlaceAnchorScanMatch & post_anchor,
  const double forward_alignment_translation_error_m,
  const double forward_alignment_yaw_error_rad,
  const double reverse_alignment_translation_error_m,
  const double reverse_alignment_yaw_error_rad) noexcept
{
  RoughnessPlacePairMatchEvidence evidence;
  evidence.available = pre_anchor.available && post_anchor.available;
  evidence.pre_anchor = placeAnchorEvidence(pre_anchor);
  evidence.post_anchor = placeAnchorEvidence(post_anchor);
  evidence.forward_alignment_translation_error_m =
    forward_alignment_translation_error_m;
  evidence.forward_alignment_yaw_error_rad = forward_alignment_yaw_error_rad;
  evidence.reverse_alignment_translation_error_m =
    reverse_alignment_translation_error_m;
  evidence.reverse_alignment_yaw_error_rad = reverse_alignment_yaw_error_rad;
  return evidence;
}

struct RoughnessPlaceCandidateEvaluation
{
  RoughnessPlaceCandidate candidate;
  RoughnessPlaceMatchDirection descriptor_direction{
    RoughnessPlaceMatchDirection::kForward};
  RoughnessPlaceAnchorPairing pairing;
  karto::LocalizedRangeScan * reference_pre_scan{nullptr};
  karto::LocalizedRangeScan * reference_post_scan{nullptr};
  karto::LocalizedRangeScan * query_for_reference_pre_scan{nullptr};
  karto::LocalizedRangeScan * query_for_reference_post_scan{nullptr};
  RoughnessPlaceAnchorScanMatch pre_anchor_match;
  RoughnessPlaceAnchorScanMatch post_anchor_match;
  double forward_alignment_translation_error_m{
    std::numeric_limits<double>::quiet_NaN()};
  double forward_alignment_yaw_error_rad{
    std::numeric_limits<double>::quiet_NaN()};
  double reverse_alignment_translation_error_m{
    std::numeric_limits<double>::quiet_NaN()};
  double reverse_alignment_yaw_error_rad{
    std::numeric_limits<double>::quiet_NaN()};
  RoughnessPlacePairMatchEvidence evidence;
  RoughnessPlaceConstraintDecision decision;
  RoughnessPlaceConstraintQuality quality;
};

bool graphContainsScan(
  karto::Mapper * mapper,
  const karto::LocalizedRangeScan * scan) noexcept
{
  if (!mapper || !mapper->GetGraph() || !scan) {
    return false;
  }
  const auto & vertices = mapper->GetGraph()->GetVertices();
  const auto sensor = vertices.find(scan->GetSensorName());
  if (sensor == vertices.end()) {
    return false;
  }
  const auto vertex = sensor->second.find(scan->GetStateId());
  return
    vertex != sensor->second.end() && vertex->second &&
    vertex->second->GetObject() == scan;
}

bool graphAlreadyConnects(
  karto::Mapper * mapper,
  const karto::LocalizedRangeScan * first,
  const karto::LocalizedRangeScan * second) noexcept
{
  if (!mapper || !mapper->GetGraph() || !first || !second) {
    return false;
  }
  for (auto * edge : mapper->GetGraph()->GetEdges()) {
    if (
      !edge || !edge->GetSource() || !edge->GetTarget() ||
      !edge->GetSource()->GetObject() || !edge->GetTarget()->GetObject())
    {
      continue;
    }
    const auto * source = edge->GetSource()->GetObject();
    const auto * target = edge->GetTarget()->GetObject();
    if (
      (source == first && target == second) ||
      (source == second && target == first))
    {
      return true;
    }
  }
  return false;
}

karto::Matrix3 makeRoughnessPlaceConstraintCovariance(
  const RoughnessPlaceAnchorScanMatch & anchor,
  const RoughnessPlaceConstraintDecision & decision,
  const double pair_covariance_scale,
  const double xy_variance_floor_m2,
  const double yaw_variance_floor_rad2,
  const bool include_pair_consistency_variance,
  const double covariance_scale) noexcept
{
  karto::Matrix3 covariance;
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      covariance(row, column) = 0.0;
    }
  }
  const double scan_xy_variance = std::max(
    covarianceXyMaximumEigenvalue(anchor.forward.fine_covariance),
    covarianceXyMaximumEigenvalue(anchor.reverse.fine_covariance));
  const double scan_yaw_variance = std::max(
    anchor.forward.fine_covariance(2, 2),
    anchor.reverse.fine_covariance(2, 2));
  covariance(0, 0) = scan_xy_variance + xy_variance_floor_m2;
  covariance(1, 1) = covariance(0, 0);
  covariance(2, 2) = scan_yaw_variance + yaw_variance_floor_rad2;
  if (include_pair_consistency_variance) {
    covariance(0, 0) += pair_covariance_scale *
      decision.maximum_pair_alignment_translation_error_m *
      decision.maximum_pair_alignment_translation_error_m;
    covariance(1, 1) = covariance(0, 0);
    covariance(2, 2) += pair_covariance_scale *
      decision.maximum_pair_alignment_yaw_error_rad *
      decision.maximum_pair_alignment_yaw_error_rad;
  }
  covariance(0, 0) *= covariance_scale;
  covariance(1, 1) *= covariance_scale;
  covariance(2, 2) *= covariance_scale;
  return covariance;
}

struct RoughnessPlaceConstraintResidual
{
  double x_m{std::numeric_limits<double>::quiet_NaN()};
  double y_m{std::numeric_limits<double>::quiet_NaN()};
  double translation_m{std::numeric_limits<double>::quiet_NaN()};
  double yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double covariance_weighted_squared{
    std::numeric_limits<double>::quiet_NaN()};
};

RoughnessPlaceConstraintResidual roughnessPlaceConstraintResidual(
  const karto::LocalizedRangeScan * source,
  const karto::LocalizedRangeScan * target,
  const karto::Pose2 & measurement,
  const karto::Matrix3 & covariance) noexcept
{
  RoughnessPlaceConstraintResidual result;
  if (!source || !target || !finitePose(measurement) || !finiteMatrix(covariance)) {
    return result;
  }
  const karto::Pose2 predicted = relativePose(
    source->GetCorrectedPose(), target->GetCorrectedPose());
  const double residual[3] = {
    predicted.GetX() - measurement.GetX(),
    predicted.GetY() - measurement.GetY(),
    normalizeAngleValue(predicted.GetHeading() - measurement.GetHeading()),
  };
  result.x_m = residual[0];
  result.y_m = residual[1];
  result.translation_m = std::hypot(residual[0], residual[1]);
  result.yaw_rad = std::fabs(residual[2]);

  karto::Matrix3 regularized = covariance;
  regularized(0, 0) += 1.0e-9;
  regularized(1, 1) += 1.0e-9;
  regularized(2, 2) += 1.0e-9;
  karto::Matrix3 information;
  if (!regularized.InverseFast(information, 1.0e-14)) {
    return result;
  }
  double weighted_squared = 0.0;
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      weighted_squared +=
        residual[row] * information(row, column) * residual[column];
    }
  }
  if (std::isfinite(weighted_squared) && weighted_squared >= 0.0) {
    result.covariance_weighted_squared = weighted_squared;
  }
  return result;
}

struct RoughnessPlacePendingConstraint
{
  const char * anchor_role{"none"};
  karto::LocalizedRangeScan * source_scan{nullptr};
  karto::LocalizedRangeScan * target_scan{nullptr};
  const RoughnessPlaceAnchorScanMatch * anchor_match{nullptr};
  karto::Pose2 measurement{unavailablePose()};
  karto::Matrix3 covariance;
  karto::Matrix3 final_covariance;
  RoughnessPlaceConstraintResidual pre_residual;
  RoughnessPlaceConstraintResidual post_residual;
  std::int64_t edge_index{-1};
  bool injected{false};

  RoughnessPlacePendingConstraint()
  {
    setMatrixUnavailable(covariance);
    setMatrixUnavailable(final_covariance);
  }
};

std::string logToken(std::string value)
{
  std::replace_if(
    value.begin(), value.end(),
    [](const unsigned char character) {
      return std::isspace(character) != 0 || character == ',';
    },
    '_');
  return value.empty() ? "none" : value;
}

}  // namespace

RoughnessAwareSlamToolbox::RoughnessAwareSlamToolbox(rclcpp::NodeOptions options)
: slam_toolbox::SlamToolbox(options)
{
  declare_parameter("enable_roughness_innovation_gate", false);
  declare_parameter("roughness_innovation_mode", "observe");
  declare_parameter("roughness_imu_topic", "/imu/data");
  declare_parameter("roughness_window_before_s", 0.10);
  declare_parameter("roughness_window_after_s", 0.10);
  declare_parameter("roughness_pending_timeout_s", 0.50);
  declare_parameter("roughness_imu_buffer_duration_s", 2.0);
  declare_parameter("roughness_imu_epoch_reset_threshold_s", 1.0);
  declare_parameter("roughness_score_gate", 2.75);
  declare_parameter("roughness_min_rough_samples", 3);
  declare_parameter("roughness_min_rough_duration_s", 0.02);
  declare_parameter("roughness_tilt_threshold_deg", 2.0);
  declare_parameter("roughness_gyro_xy_threshold_rad_s", 0.05);
  declare_parameter("roughness_accel_norm_deviation_threshold_m_s2", 0.35);
  declare_parameter("roughness_tilt_rate_threshold_deg_s", 5.0);
  declare_parameter("roughness_accel_norm_rate_threshold_m_s3", 80.0);
  declare_parameter("roughness_translation_innovation_gate_m", 0.04);
  declare_parameter("roughness_yaw_innovation_gate_deg", 3.0);
  declare_parameter("roughness_enable_near_chain_gate", false);
  declare_parameter("roughness_near_chain_recovery_duration_s", 2.0);
  declare_parameter("roughness_near_chain_loop_closing_cooldown_s", 0.0);
  declare_parameter("roughness_near_chain_min_rough_samples", 1);
  declare_parameter("roughness_near_chain_gated_min_response", 1.1);
  declare_parameter("roughness_constraint_mode", "");
  declare_parameter("roughness_enable_near_chain_candidate_observer", false);
  declare_parameter("roughness_enable_loop_closure_candidate_observer", false);
  declare_parameter("roughness_near_chain_observer_min_parity_candidates", 20);
  declare_parameter("roughness_adaptive_memory_trigger_score", -1.0);
  declare_parameter("roughness_adaptive_memory_saturation_score", 5.5);
  declare_parameter("roughness_adaptive_memory_recovery_score", 1.0);
  declare_parameter("roughness_adaptive_memory_min_duration_s", 0.5);
  declare_parameter("roughness_adaptive_memory_max_duration_s", 3.5);
  declare_parameter("roughness_adaptive_memory_min_distance_m", 0.10);
  declare_parameter("roughness_adaptive_memory_max_distance_m", 0.55);
  declare_parameter("roughness_adaptive_memory_recovery_time_constant_s", 0.5);
  declare_parameter("roughness_near_chain_quality_calibrated", false);
  declare_parameter("roughness_near_chain_quality_response_bad", 0.08);
  declare_parameter("roughness_near_chain_quality_response_good", 0.30);
  declare_parameter(
    "roughness_near_chain_quality_translation_good_m", 0.02);
  declare_parameter(
    "roughness_near_chain_quality_translation_bad_m", 0.10);
  declare_parameter("roughness_near_chain_quality_yaw_good_deg", 1.0);
  declare_parameter("roughness_near_chain_quality_yaw_bad_deg", 5.0);
  declare_parameter("roughness_near_chain_quality_xy_covariance_good_m2", 0.0025);
  declare_parameter("roughness_near_chain_quality_xy_covariance_bad_m2", 0.04);
  declare_parameter("roughness_near_chain_quality_yaw_covariance_good_rad2", 0.0003046174);
  declare_parameter("roughness_near_chain_quality_yaw_covariance_bad_rad2", 0.0304617420);
  declare_parameter("roughness_adaptive_downweight_risk", 0.35);
  declare_parameter("roughness_adaptive_reject_risk", 0.80);
  declare_parameter("roughness_adaptive_minimum_information_scale", 0.10);
  declare_parameter("roughness_adaptive_dynamic_threshold_max", 1.0);
  declare_parameter("roughness_enable_rotation_consistency_guard", false);
  declare_parameter("roughness_rotation_history_size", 64);
  declare_parameter("roughness_rotation_minimum_history_batches", 8);
  declare_parameter("roughness_rotation_outlier_sigma", 6.0);
  declare_parameter("roughness_rotation_extreme_yaw_sigma", 12.0);
  declare_parameter("roughness_rotation_yaw_scale_floor_deg", 0.4);
  declare_parameter("roughness_rotation_translation_scale_floor_m", 0.03);
  declare_parameter(
    "roughness_rotation_weighted_disagreement_scale_floor", 0.25);
  declare_parameter("roughness_rotation_chain_overlap_trigger", 0.75);
  declare_parameter("roughness_rotation_maximum_repeat_state_gap", 2);
  declare_parameter("roughness_rotation_stable_recovery_batches", 3);
  declare_parameter(
    "roughness_disable_loop_closing_during_near_chain_gate", false);
  declare_parameter("roughness_enable_spatial_memory_gate", false);
  declare_parameter("roughness_spatial_memory_use_pose_graph_anchors", false);
  declare_parameter("roughness_spatial_memory_radius_m", 0.6);
  declare_parameter(
    "roughness_spatial_memory_min_revisit_separation_s", 20.0);
  declare_parameter(
    "roughness_spatial_memory_min_heading_delta_deg", 126.0);
  declare_parameter("roughness_spatial_memory_merge_radius_m", 0.10);
  declare_parameter("roughness_spatial_memory_max_zones", 200);
  declare_parameter("roughness_enable_spatial_passage_hysteresis", false);
  declare_parameter("roughness_spatial_passage_radius_m", 0.55);
  declare_parameter("roughness_spatial_passage_max_age_s", 3.5);
  declare_parameter("roughness_enable_post_confirmation", true);
  declare_parameter("roughness_post_confirmation_max_age_s", 2.0);
  declare_parameter("roughness_post_confirmation_min_base_scans", 8);
  declare_parameter("roughness_post_confirmation_max_base_scans", 10);
  declare_parameter("roughness_post_confirmation_min_response", 0.30);
  declare_parameter("roughness_post_confirmation_min_response_gain", 0.05);
  declare_parameter(
    "roughness_post_confirmation_min_covariance_improvement_ratio", 1.5);
  declare_parameter(
    "roughness_post_confirmation_translation_disagreement_gate_m", 0.03);
  declare_parameter(
    "roughness_post_confirmation_yaw_disagreement_gate_deg", 1.0);
  declare_parameter("roughness_enable_place_recognition_observer", true);
  declare_parameter("roughness_enable_place_scan_match_observer", false);
  declare_parameter("roughness_place_evaluate_both_directions", false);
  declare_parameter("roughness_place_enable_reference_chain_observer", false);
  declare_parameter("roughness_place_reference_chain_scan_count", 5);
  declare_parameter("roughness_place_reference_chain_max_distance_m", 1.0);
  declare_parameter("roughness_place_reference_chain_max_state_gap", 5);
  declare_parameter("roughness_place_reference_chain_max_time_gap_s", 5.0);
  declare_parameter("roughness_place_descriptor_bins", 16);
  declare_parameter("roughness_place_max_observations_per_event", 512);
  declare_parameter("roughness_place_max_events", 200);
  declare_parameter("roughness_place_candidate_top_k", 5);
  declare_parameter("roughness_place_min_revisit_separation_s", 20.0);
  declare_parameter("roughness_place_min_state_separation", 20);
  declare_parameter("roughness_place_constraint_mode", "observe");
  declare_parameter(
    "roughness_place_constraint_anchor_strategy", "best_single");
  declare_parameter("roughness_place_min_fine_response", 0.30);
  declare_parameter(
    "roughness_place_max_proposal_disagreement_translation_m", 0.30);
  declare_parameter("roughness_place_max_proposal_disagreement_yaw_deg", 5.0);
  declare_parameter("roughness_place_max_fine_xy_covariance_m2", 0.04);
  declare_parameter("roughness_place_max_fine_yaw_covariance_deg2", 25.0);
  declare_parameter("roughness_place_max_bidirectional_translation_error_m", 0.10);
  declare_parameter("roughness_place_max_bidirectional_yaw_error_deg", 2.0);
  declare_parameter("roughness_place_max_pair_alignment_translation_error_m", 0.50);
  declare_parameter("roughness_place_max_pair_alignment_yaw_error_deg", 3.0);
  declare_parameter("roughness_place_constraint_pair_covariance_scale", 1.0);
  declare_parameter("roughness_place_constraint_group_covariance_scale", 1.0);
  declare_parameter("roughness_place_constraint_xy_variance_floor_m2", 0.01);
  declare_parameter("roughness_place_constraint_yaw_variance_floor_deg2", 1.0);
  declare_parameter("roughness_diagnostics_period_s", 10.0);
  declare_parameter("roughness_max_pending_scans", 2000);
  declare_parameter("roughness_imu_qos_reliable", false);
  declare_parameter("roughness_imu_qos_depth", 2000);
  declare_parameter("roughness_scan_qos_reliable", false);
  declare_parameter("roughness_scan_qos_depth", 5);
  declare_parameter("enable_roughness_scan_conditioning", false);
  declare_parameter("roughness_scan_conditioning_mode", "observe");
  declare_parameter("roughness_scan_conditioning_use_nearest_imu_sample", true);
  declare_parameter("roughness_scan_conditioning_max_scan_imu_dt_s", 0.2);
  declare_parameter("roughness_scan_conditioning_hold_duration_s", 0.1);
  declare_parameter("roughness_scan_conditioning_hard_hold_duration_s", 0.0);
  declare_parameter("roughness_scan_conditioning_hard_score_gate", 3.0);
  declare_parameter("roughness_scan_conditioning_throttle_min_publish_period_s", 0.5);
  declare_parameter("roughness_scan_conditioning_throttle_drop_hard", true);
  declare_parameter("roughness_scan_conditioning_max_continuous_drop_duration_s", 0.0);
  declare_parameter("roughness_scan_conditioning_stabilize_max_source_age_s", 0.75);
  declare_parameter("roughness_scan_conditioning_stabilize_max_event_duration_s", 0.5);
  declare_parameter("roughness_scan_conditioning_stabilize_min_publish_period_s", 0.1);
  declare_parameter("roughness_scan_conditioning_stabilize_max_translation_m", 0.2);
  declare_parameter("roughness_scan_conditioning_stabilize_max_yaw_rad", 0.15);
  declare_parameter("roughness_scan_conditioning_stabilize_max_roll_pitch_rad", 0.05);
  declare_parameter("roughness_scan_conditioning_stabilize_min_cached_valid_ratio", 0.2);
  declare_parameter("roughness_scan_conditioning_stabilize_min_output_valid_ratio", 0.15);
  declare_parameter("roughness_scan_conditioning_stabilize_min_output_valid_beams", 20);
  declare_parameter("roughness_scan_conditioning_stabilize_fallback_mode", "drop");
  declare_parameter("roughness_scan_conditioning_residual_max_source_age_s", 0.75);
  declare_parameter("roughness_scan_conditioning_residual_max_translation_m", 0.2);
  declare_parameter("roughness_scan_conditioning_residual_max_yaw_rad", 0.15);
  declare_parameter("roughness_scan_conditioning_residual_max_roll_pitch_rad", 0.05);
  declare_parameter("roughness_scan_conditioning_residual_threshold_m", 0.08);
  declare_parameter("roughness_scan_conditioning_residual_neighbor_bins", 1);
  declare_parameter("roughness_scan_conditioning_residual_min_cached_valid_ratio", 0.2);
  declare_parameter("roughness_scan_conditioning_residual_min_overlap_ratio", 0.35);
  declare_parameter("roughness_scan_conditioning_residual_min_overlap_beams", 100);
  declare_parameter("roughness_scan_conditioning_residual_min_output_valid_ratio", 0.2);
  declare_parameter("roughness_scan_conditioning_residual_min_output_valid_beams", 100);
  declare_parameter("roughness_scan_conditioning_residual_direction", "closer");
  declare_parameter("roughness_scan_conditioning_residual_max_removed_ratio", 0.03);
  declare_parameter("roughness_scan_conditioning_residual_promote_filtered_reference", false);
  declare_parameter("roughness_scan_conditioning_residual_fallback_mode", "observe");
  declare_parameter("roughness_scan_conditioning_quarantine_recovery_scan_count", 5);
  declare_parameter("roughness_scan_conditioning_quarantine_max_scan_gap_s", 0.75);
  declare_parameter("roughness_scan_conditioning_quarantine_min_stable_duration_s", 0.16);
  declare_parameter("roughness_scan_conditioning_quarantine_fault_after_s", 3.0);
  declare_parameter("roughness_scan_conditioning_quarantine_fail_open_on_imu_fault", true);
  declare_parameter("roughness_scan_conditioning_quarantine_min_valid_ratio", 0.2);
  declare_parameter("roughness_experimental_reject_scan_stamps_ns", "");
}

RoughnessAwareSlamToolbox::~RoughnessAwareSlamToolbox()
{
  shutdown();
}

void RoughnessAwareSlamToolbox::shutdown() noexcept
{
  if (shutdown_started_.exchange(true)) {
    return;
  }

  pending_timer_.reset();
  clear_queue_service_.reset();
  finalize_pending_service_.reset();
  imu_sub_.reset();
  scan_filter_.reset();
  scan_filter_sub_.reset();

  stop_scan_worker_.store(true);
  scan_queue_cv_.notify_all();
  if (scan_worker_.joinable()) {
    try {
      scan_worker_.join();
    } catch (...) {
      return;
    }
  }

  try {
    for (auto & thread : threads_) {
      if (thread && thread->joinable()) {
        thread->join();
      }
    }
    threads_.clear();

    // These helpers and the solver retain shared ownership of this node.
    // Tear down the Karto state while its process-wide SensorManager is alive,
    // then release those ownership cycles before main drops its final handle.
    smapper_.reset();
    dataset_.reset();
    closure_assistant_.reset();
    map_saver_.reset();
    pose_helper_.reset();
    laser_assistant_.reset();
    scan_holder_.reset();
    solver_.reset();
  } catch (...) {
    // Destructors must not let shutdown failures escape during stack unwinding.
  }
}

void RoughnessAwareSlamToolbox::configureScanSubscription()
{
  rmw_qos_profile_t scan_qos = rmw_qos_profile_sensor_data;
  scan_qos.depth = scan_qos_depth_;
  if (scan_qos_reliable_) {
    scan_qos.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
  }

  scan_filter_.reset();
  scan_filter_sub_.reset();
  scan_filter_sub_ =
    std::make_unique<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>(
    shared_from_this().get(), scan_topic_, scan_qos);
  scan_filter_ =
    std::make_unique<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>>(
    *scan_filter_sub_, *tf_, odom_frame_, scan_queue_size_, shared_from_this(),
    tf2::durationFromSec(transform_timeout_.seconds()));
  scan_filter_->registerCallback(
    std::bind(
      &RoughnessAwareSlamToolbox::laserCallback,
      this,
      std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(),
    "LaserScan subscription QoS: reliability=%s depth=%zu",
    scan_qos_reliable_ ? "reliable" : "best_effort",
    scan_qos_depth_);
}

void RoughnessAwareSlamToolbox::configure()
{
  if (scan_worker_.joinable()) {
    throw std::logic_error("RoughnessAwareSlamToolbox is already configured");
  }
  get_parameter("roughness_scan_qos_reliable", scan_qos_reliable_);
  int scan_qos_depth = 5;
  get_parameter("roughness_scan_qos_depth", scan_qos_depth);
  scan_qos_depth_ = static_cast<std::size_t>(std::max(scan_qos_depth, 1));
  get_parameter("roughness_imu_qos_reliable", imu_qos_reliable_);
  int imu_qos_depth = 2000;
  get_parameter("roughness_imu_qos_depth", imu_qos_depth);
  imu_qos_depth_ = static_cast<std::size_t>(std::max(imu_qos_depth, 1));
  slam_toolbox::SlamToolbox::configure();
  configureScanSubscription();

  get_parameter("enable_roughness_innovation_gate", enabled_);
  get_parameter("roughness_innovation_mode", mode_);
  get_parameter("roughness_imu_topic", imu_topic_);
  get_parameter("roughness_window_before_s", window_before_s_);
  get_parameter("roughness_window_after_s", window_after_s_);
  get_parameter("roughness_pending_timeout_s", pending_timeout_s_);
  get_parameter("roughness_imu_buffer_duration_s", imu_buffer_duration_s_);
  get_parameter("roughness_imu_epoch_reset_threshold_s", imu_epoch_reset_threshold_s_);
  get_parameter("roughness_score_gate", roughness_score_gate_);
  int min_rough_samples = 3;
  get_parameter("roughness_min_rough_samples", min_rough_samples);
  get_parameter("roughness_min_rough_duration_s", min_rough_duration_s_);
  get_parameter("roughness_tilt_threshold_deg", roughness_thresholds_.tilt_deg);
  get_parameter(
    "roughness_gyro_xy_threshold_rad_s",
    roughness_thresholds_.gyro_xy_rad_s);
  get_parameter(
    "roughness_accel_norm_deviation_threshold_m_s2",
    roughness_thresholds_.accel_norm_deviation_m_s2);
  get_parameter(
    "roughness_tilt_rate_threshold_deg_s",
    roughness_thresholds_.tilt_rate_deg_s);
  get_parameter(
    "roughness_accel_norm_rate_threshold_m_s3",
    roughness_thresholds_.accel_norm_rate_m_s3);
  get_parameter(
    "roughness_translation_innovation_gate_m",
    translation_innovation_gate_m_);
  double yaw_innovation_gate_deg = 3.0;
  get_parameter("roughness_yaw_innovation_gate_deg", yaw_innovation_gate_deg);
  get_parameter(
    "roughness_enable_near_chain_gate",
    enable_near_chain_gate_);
  get_parameter(
    "roughness_near_chain_recovery_duration_s",
    near_chain_recovery_duration_s_);
  get_parameter(
    "roughness_near_chain_loop_closing_cooldown_s",
    near_chain_loop_closing_cooldown_s_);
  int near_chain_min_rough_samples = 1;
  get_parameter(
    "roughness_near_chain_min_rough_samples",
    near_chain_min_rough_samples);
  get_parameter(
    "roughness_near_chain_gated_min_response",
    near_chain_gated_min_response_);
  get_parameter("roughness_constraint_mode", roughness_constraint_mode_name_);
  get_parameter(
    "roughness_enable_near_chain_candidate_observer",
    enable_near_chain_candidate_observer_);
  get_parameter(
    "roughness_enable_loop_closure_candidate_observer",
    enable_loop_closure_candidate_observer_);
  int near_chain_observer_min_parity_candidates = 20;
  get_parameter(
    "roughness_near_chain_observer_min_parity_candidates",
    near_chain_observer_min_parity_candidates);
  double adaptive_memory_trigger_score = -1.0;
  get_parameter(
    "roughness_adaptive_memory_trigger_score",
    adaptive_memory_trigger_score);
  get_parameter(
    "roughness_adaptive_memory_saturation_score",
    adaptive_memory_config_.saturation_score);
  get_parameter(
    "roughness_adaptive_memory_recovery_score",
    adaptive_memory_config_.recovery_score);
  get_parameter(
    "roughness_adaptive_memory_min_duration_s",
    adaptive_memory_config_.minimum_duration_s);
  get_parameter(
    "roughness_adaptive_memory_max_duration_s",
    adaptive_memory_config_.maximum_duration_s);
  get_parameter(
    "roughness_adaptive_memory_min_distance_m",
    adaptive_memory_config_.minimum_distance_m);
  get_parameter(
    "roughness_adaptive_memory_max_distance_m",
    adaptive_memory_config_.maximum_distance_m);
  get_parameter(
    "roughness_adaptive_memory_recovery_time_constant_s",
    adaptive_memory_config_.stable_recovery_time_constant_s);
  get_parameter(
    "roughness_near_chain_quality_calibrated",
    near_chain_quality_config_.calibrated);
  get_parameter(
    "roughness_near_chain_quality_response_bad",
    near_chain_quality_config_.response_bad);
  get_parameter(
    "roughness_near_chain_quality_response_good",
    near_chain_quality_config_.response_good);
  get_parameter(
    "roughness_near_chain_quality_translation_good_m",
    near_chain_quality_config_.translation_disagreement_good_m);
  get_parameter(
    "roughness_near_chain_quality_translation_bad_m",
    near_chain_quality_config_.translation_disagreement_bad_m);
  double near_chain_quality_yaw_good_deg = 1.0;
  double near_chain_quality_yaw_bad_deg = 5.0;
  get_parameter(
    "roughness_near_chain_quality_yaw_good_deg",
    near_chain_quality_yaw_good_deg);
  get_parameter(
    "roughness_near_chain_quality_yaw_bad_deg",
    near_chain_quality_yaw_bad_deg);
  get_parameter(
    "roughness_near_chain_quality_xy_covariance_good_m2",
    near_chain_quality_config_.xy_covariance_good_m2);
  get_parameter(
    "roughness_near_chain_quality_xy_covariance_bad_m2",
    near_chain_quality_config_.xy_covariance_bad_m2);
  get_parameter(
    "roughness_near_chain_quality_yaw_covariance_good_rad2",
    near_chain_quality_config_.yaw_covariance_good_rad2);
  get_parameter(
    "roughness_near_chain_quality_yaw_covariance_bad_rad2",
    near_chain_quality_config_.yaw_covariance_bad_rad2);
  get_parameter(
    "roughness_adaptive_downweight_risk",
    adaptive_constraint_policy_config_.downweight_risk);
  get_parameter(
    "roughness_adaptive_reject_risk",
    adaptive_constraint_policy_config_.reject_risk);
  get_parameter(
    "roughness_adaptive_minimum_information_scale",
    adaptive_constraint_policy_config_.minimum_information_scale);
  get_parameter(
    "roughness_adaptive_dynamic_threshold_max",
    adaptive_dynamic_threshold_max_);
  get_parameter(
    "roughness_enable_rotation_consistency_guard",
    rotation_consistency_config_.enabled);
  int rotation_history_size = 64;
  int rotation_minimum_history_batches = 8;
  int rotation_maximum_repeat_state_gap = 2;
  int rotation_stable_recovery_batches = 3;
  get_parameter("roughness_rotation_history_size", rotation_history_size);
  get_parameter(
    "roughness_rotation_minimum_history_batches",
    rotation_minimum_history_batches);
  get_parameter(
    "roughness_rotation_outlier_sigma",
    rotation_consistency_config_.outlier_sigma);
  get_parameter(
    "roughness_rotation_extreme_yaw_sigma",
    rotation_consistency_config_.extreme_yaw_sigma);
  double rotation_yaw_scale_floor_deg = 0.4;
  get_parameter(
    "roughness_rotation_yaw_scale_floor_deg",
    rotation_yaw_scale_floor_deg);
  get_parameter(
    "roughness_rotation_translation_scale_floor_m",
    rotation_consistency_config_.translation_scale_floor_m);
  get_parameter(
    "roughness_rotation_weighted_disagreement_scale_floor",
    rotation_consistency_config_.weighted_disagreement_scale_floor);
  get_parameter(
    "roughness_rotation_chain_overlap_trigger",
    rotation_consistency_config_.chain_overlap_trigger);
  get_parameter(
    "roughness_rotation_maximum_repeat_state_gap",
    rotation_maximum_repeat_state_gap);
  get_parameter(
    "roughness_rotation_stable_recovery_batches",
    rotation_stable_recovery_batches);
  get_parameter(
    "roughness_disable_loop_closing_during_near_chain_gate",
    disable_loop_closing_during_near_chain_gate_);
  get_parameter(
    "roughness_enable_spatial_memory_gate",
    enable_spatial_memory_gate_);
  get_parameter(
    "roughness_spatial_memory_use_pose_graph_anchors",
    spatial_memory_use_pose_graph_anchors_);
  get_parameter(
    "roughness_spatial_memory_radius_m",
    spatial_memory_radius_m_);
  get_parameter(
    "roughness_spatial_memory_min_revisit_separation_s",
    spatial_memory_min_revisit_separation_s_);
  double spatial_memory_min_heading_delta_deg = 126.0;
  get_parameter(
    "roughness_spatial_memory_min_heading_delta_deg",
    spatial_memory_min_heading_delta_deg);
  get_parameter(
    "roughness_spatial_memory_merge_radius_m",
    spatial_memory_merge_radius_m_);
  int spatial_memory_max_zones = 200;
  get_parameter(
    "roughness_spatial_memory_max_zones",
    spatial_memory_max_zones);
  get_parameter(
    "roughness_enable_spatial_passage_hysteresis",
    enable_spatial_passage_hysteresis_);
  get_parameter(
    "roughness_spatial_passage_radius_m",
    spatial_passage_radius_m_);
  get_parameter(
    "roughness_spatial_passage_max_age_s",
    spatial_passage_max_age_s_);
  get_parameter(
    "roughness_enable_post_confirmation",
    enable_post_confirmation_);
  get_parameter(
    "roughness_post_confirmation_max_age_s",
    post_confirmation_max_age_s_);
  int post_confirmation_min_base_scans = 8;
  get_parameter(
    "roughness_post_confirmation_min_base_scans",
    post_confirmation_min_base_scans);
  int post_confirmation_max_base_scans = 10;
  get_parameter(
    "roughness_post_confirmation_max_base_scans",
    post_confirmation_max_base_scans);
  get_parameter(
    "roughness_post_confirmation_min_response",
    post_confirmation_min_response_);
  get_parameter(
    "roughness_post_confirmation_min_response_gain",
    post_confirmation_min_response_gain_);
  get_parameter(
    "roughness_post_confirmation_min_covariance_improvement_ratio",
    post_confirmation_min_covariance_improvement_ratio_);
  get_parameter(
    "roughness_post_confirmation_translation_disagreement_gate_m",
    post_confirmation_translation_disagreement_gate_m_);
  double post_confirmation_yaw_disagreement_gate_deg = 1.0;
  get_parameter(
    "roughness_post_confirmation_yaw_disagreement_gate_deg",
    post_confirmation_yaw_disagreement_gate_deg);
  get_parameter(
    "roughness_enable_place_recognition_observer",
    enable_roughness_place_recognition_observer_);
  get_parameter(
    "roughness_enable_place_scan_match_observer",
    enable_roughness_place_scan_match_observer_);
  get_parameter(
    "roughness_place_evaluate_both_directions",
    roughness_place_evaluate_both_directions_);
  get_parameter(
    "roughness_place_enable_reference_chain_observer",
    roughness_place_enable_reference_chain_observer_);
  int roughness_place_reference_chain_scan_count = 5;
  get_parameter(
    "roughness_place_reference_chain_scan_count",
    roughness_place_reference_chain_scan_count);
  get_parameter(
    "roughness_place_reference_chain_max_distance_m",
    roughness_place_scan_chain_config_.maximum_anchor_distance_m);
  int roughness_place_reference_chain_max_state_gap = 5;
  get_parameter(
    "roughness_place_reference_chain_max_state_gap",
    roughness_place_reference_chain_max_state_gap);
  get_parameter(
    "roughness_place_reference_chain_max_time_gap_s",
    roughness_place_scan_chain_config_.maximum_anchor_time_gap_s);
  int roughness_place_descriptor_bins = 16;
  get_parameter(
    "roughness_place_descriptor_bins",
    roughness_place_descriptor_bins);
  int roughness_place_max_observations_per_event = 512;
  get_parameter(
    "roughness_place_max_observations_per_event",
    roughness_place_max_observations_per_event);
  int roughness_place_max_events = 200;
  get_parameter(
    "roughness_place_max_events",
    roughness_place_max_events);
  int roughness_place_candidate_top_k = 5;
  get_parameter(
    "roughness_place_candidate_top_k",
    roughness_place_candidate_top_k);
  get_parameter(
    "roughness_place_min_revisit_separation_s",
    roughness_place_min_revisit_separation_s_);
  int roughness_place_min_state_separation = 20;
  get_parameter(
    "roughness_place_min_state_separation",
    roughness_place_min_state_separation);
  get_parameter(
    "roughness_place_constraint_mode",
    roughness_place_constraint_mode_);
  get_parameter(
    "roughness_place_constraint_anchor_strategy",
    roughness_place_constraint_anchor_strategy_);
  get_parameter(
    "roughness_place_min_fine_response",
    roughness_place_constraint_thresholds_.minimum_fine_response);
  get_parameter(
    "roughness_place_max_proposal_disagreement_translation_m",
    roughness_place_constraint_thresholds_.
    maximum_proposal_disagreement_translation_m);
  double roughness_place_max_proposal_disagreement_yaw_deg = 5.0;
  get_parameter(
    "roughness_place_max_proposal_disagreement_yaw_deg",
    roughness_place_max_proposal_disagreement_yaw_deg);
  get_parameter(
    "roughness_place_max_fine_xy_covariance_m2",
    roughness_place_constraint_thresholds_.maximum_fine_xy_covariance_m2);
  double roughness_place_max_fine_yaw_covariance_deg2 = 25.0;
  get_parameter(
    "roughness_place_max_fine_yaw_covariance_deg2",
    roughness_place_max_fine_yaw_covariance_deg2);
  get_parameter(
    "roughness_place_max_bidirectional_translation_error_m",
    roughness_place_constraint_thresholds_.
    maximum_bidirectional_translation_error_m);
  double roughness_place_max_bidirectional_yaw_error_deg = 2.0;
  get_parameter(
    "roughness_place_max_bidirectional_yaw_error_deg",
    roughness_place_max_bidirectional_yaw_error_deg);
  get_parameter(
    "roughness_place_max_pair_alignment_translation_error_m",
    roughness_place_constraint_thresholds_.
    maximum_pair_alignment_translation_error_m);
  double roughness_place_max_pair_alignment_yaw_error_deg = 3.0;
  get_parameter(
    "roughness_place_max_pair_alignment_yaw_error_deg",
    roughness_place_max_pair_alignment_yaw_error_deg);
  get_parameter(
    "roughness_place_constraint_pair_covariance_scale",
    roughness_place_constraint_pair_covariance_scale_);
  get_parameter(
    "roughness_place_constraint_group_covariance_scale",
    roughness_place_constraint_group_covariance_scale_);
  get_parameter(
    "roughness_place_constraint_xy_variance_floor_m2",
    roughness_place_constraint_xy_variance_floor_m2_);
  double roughness_place_constraint_yaw_variance_floor_deg2 = 1.0;
  get_parameter(
    "roughness_place_constraint_yaw_variance_floor_deg2",
    roughness_place_constraint_yaw_variance_floor_deg2);
  get_parameter("roughness_diagnostics_period_s", diagnostics_period_s_);
  int max_pending_scans = 2000;
  get_parameter("roughness_max_pending_scans", max_pending_scans);
  get_parameter("enable_roughness_scan_conditioning", enable_scan_conditioning_);
  get_parameter("roughness_scan_conditioning_mode", scan_conditioning_mode_name_);
  get_parameter(
    "roughness_scan_conditioning_use_nearest_imu_sample",
    scan_conditioning_use_nearest_imu_);
  get_parameter(
    "roughness_scan_conditioning_max_scan_imu_dt_s",
    scan_conditioning_max_scan_imu_dt_s_);
  get_parameter(
    "roughness_scan_conditioning_hold_duration_s",
    scan_conditioning_hold_duration_s_);
  get_parameter(
    "roughness_scan_conditioning_hard_hold_duration_s",
    scan_conditioning_hard_hold_duration_s_);
  get_parameter(
    "roughness_scan_conditioning_hard_score_gate",
    scan_conditioning_hard_score_gate_);
  get_parameter(
    "roughness_scan_conditioning_throttle_min_publish_period_s",
    scan_conditioning_throttle_min_publish_period_s_);
  get_parameter(
    "roughness_scan_conditioning_throttle_drop_hard",
    scan_conditioning_throttle_drop_hard_);
  get_parameter(
    "roughness_scan_conditioning_max_continuous_drop_duration_s",
    scan_conditioning_max_continuous_drop_duration_s_);
  get_parameter(
    "roughness_scan_conditioning_stabilize_max_source_age_s",
    scan_conditioning_stabilize_max_source_age_s_);
  get_parameter(
    "roughness_scan_conditioning_stabilize_max_event_duration_s",
    scan_conditioning_stabilize_max_event_duration_s_);
  get_parameter(
    "roughness_scan_conditioning_stabilize_min_publish_period_s",
    scan_conditioning_stabilize_min_publish_period_s_);
  get_parameter(
    "roughness_scan_conditioning_stabilize_max_translation_m",
    scan_conditioning_stabilize_max_translation_m_);
  get_parameter(
    "roughness_scan_conditioning_stabilize_max_yaw_rad",
    scan_conditioning_stabilize_max_yaw_rad_);
  get_parameter(
    "roughness_scan_conditioning_stabilize_max_roll_pitch_rad",
    scan_conditioning_stabilize_max_roll_pitch_rad_);
  get_parameter(
    "roughness_scan_conditioning_stabilize_min_cached_valid_ratio",
    scan_conditioning_stabilize_min_cached_valid_ratio_);
  get_parameter(
    "roughness_scan_conditioning_stabilize_min_output_valid_ratio",
    scan_conditioning_stabilize_min_output_valid_ratio_);
  int conditioning_stabilize_min_output_valid_beams = 20;
  get_parameter(
    "roughness_scan_conditioning_stabilize_min_output_valid_beams",
    conditioning_stabilize_min_output_valid_beams);
  get_parameter(
    "roughness_scan_conditioning_stabilize_fallback_mode",
    scan_conditioning_stabilize_fallback_mode_);
  get_parameter(
    "roughness_scan_conditioning_residual_max_source_age_s",
    scan_conditioning_residual_max_source_age_s_);
  get_parameter(
    "roughness_scan_conditioning_residual_max_translation_m",
    scan_conditioning_residual_max_translation_m_);
  get_parameter(
    "roughness_scan_conditioning_residual_max_yaw_rad",
    scan_conditioning_residual_max_yaw_rad_);
  get_parameter(
    "roughness_scan_conditioning_residual_max_roll_pitch_rad",
    scan_conditioning_residual_max_roll_pitch_rad_);
  get_parameter(
    "roughness_scan_conditioning_residual_threshold_m",
    scan_conditioning_residual_threshold_m_);
  int conditioning_residual_neighbor_bins = 1;
  get_parameter(
    "roughness_scan_conditioning_residual_neighbor_bins",
    conditioning_residual_neighbor_bins);
  get_parameter(
    "roughness_scan_conditioning_residual_min_cached_valid_ratio",
    scan_conditioning_residual_min_cached_valid_ratio_);
  get_parameter(
    "roughness_scan_conditioning_residual_min_overlap_ratio",
    scan_conditioning_residual_min_overlap_ratio_);
  int conditioning_residual_min_overlap_beams = 100;
  get_parameter(
    "roughness_scan_conditioning_residual_min_overlap_beams",
    conditioning_residual_min_overlap_beams);
  get_parameter(
    "roughness_scan_conditioning_residual_min_output_valid_ratio",
    scan_conditioning_residual_min_output_valid_ratio_);
  int conditioning_residual_min_output_valid_beams = 100;
  get_parameter(
    "roughness_scan_conditioning_residual_min_output_valid_beams",
    conditioning_residual_min_output_valid_beams);
  get_parameter(
    "roughness_scan_conditioning_residual_direction",
    scan_conditioning_residual_direction_name_);
  get_parameter(
    "roughness_scan_conditioning_residual_max_removed_ratio",
    scan_conditioning_residual_max_removed_ratio_);
  get_parameter(
    "roughness_scan_conditioning_residual_promote_filtered_reference",
    scan_conditioning_residual_promote_filtered_reference_);
  get_parameter(
    "roughness_scan_conditioning_residual_fallback_mode",
    scan_conditioning_residual_fallback_mode_);
  int conditioning_quarantine_recovery_scan_count = 5;
  get_parameter(
    "roughness_scan_conditioning_quarantine_recovery_scan_count",
    conditioning_quarantine_recovery_scan_count);
  get_parameter(
    "roughness_scan_conditioning_quarantine_max_scan_gap_s",
    scan_conditioning_recovery_config_.max_scan_gap_s);
  get_parameter(
    "roughness_scan_conditioning_quarantine_min_stable_duration_s",
    scan_conditioning_recovery_config_.min_stable_duration_s);
  get_parameter(
    "roughness_scan_conditioning_quarantine_fault_after_s",
    scan_conditioning_recovery_config_.fault_after_s);
  get_parameter(
    "roughness_scan_conditioning_quarantine_fail_open_on_imu_fault",
    scan_conditioning_quarantine_fail_open_on_imu_fault_);
  get_parameter(
    "roughness_scan_conditioning_quarantine_min_valid_ratio",
    scan_conditioning_quarantine_min_valid_ratio_);
  std::string experimental_reject_stamps;
  get_parameter(
    "roughness_experimental_reject_scan_stamps_ns",
    experimental_reject_stamps);

  std::transform(
    mode_.begin(), mode_.end(), mode_.begin(),
    [](const unsigned char value) {return static_cast<char>(std::tolower(value));});
  if (mode_ != "observe") {
    RCLCPP_WARN(
      get_logger(),
      "roughness_innovation_mode '%s' is not permitted for online rejection; "
      "using observe. Use exact experimental reject stamps only for bounded bag tests.",
      mode_.c_str());
    mode_ = "observe";
  }
  window_before_s_ = std::max(window_before_s_, 0.0);
  window_after_s_ = std::max(window_after_s_, 0.0);
  pending_timeout_s_ = std::max(pending_timeout_s_, window_after_s_);
  imu_buffer_duration_s_ = std::max(
    imu_buffer_duration_s_,
    window_before_s_ + window_after_s_ + pending_timeout_s_ + 0.5);
  imu_epoch_reset_threshold_s_ = std::max(imu_epoch_reset_threshold_s_, 0.0);
  roughness_score_gate_ = std::max(roughness_score_gate_, 0.0);
  min_rough_samples_ = static_cast<std::size_t>(std::max(min_rough_samples, 1));
  min_rough_duration_s_ = std::max(min_rough_duration_s_, 0.0);
  translation_innovation_gate_m_ = std::max(translation_innovation_gate_m_, 0.0);
  yaw_innovation_gate_rad_ =
    std::max(yaw_innovation_gate_deg, 0.0) * kDegreesToRadians;

  std::transform(
    roughness_constraint_mode_name_.begin(),
    roughness_constraint_mode_name_.end(),
    roughness_constraint_mode_name_.begin(),
    [](const unsigned char value) {
      return static_cast<char>(std::tolower(value));
    });
  if (roughness_constraint_mode_name_.empty()) {
    roughness_constraint_mode_ = RoughnessConstraintMode::kInterventionOff;
    roughness_constraint_mode_name_ = roughnessConstraintModeName(
      roughness_constraint_mode_);
    if (enable_near_chain_gate_) {
      RCLCPP_WARN(
        get_logger(),
        "roughness_enable_near_chain_gate no longer implicitly enables the "
        "legacy fixed-memory rejection. Set roughness_constraint_mode="
        "legacy_fixed_memory explicitly for that ablation.");
    }
  } else if (!parseRoughnessConstraintMode(
      roughness_constraint_mode_name_, roughness_constraint_mode_))
  {
    throw std::invalid_argument(
            "Unsupported roughness_constraint_mode: " +
            roughness_constraint_mode_name_);
  }
  if (
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kLegacyFixedMemory)
  {
    enable_near_chain_gate_ = true;
  }
  const bool adaptive_constraint_mode =
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveMemoryObserve ||
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveMemoryOnly ||
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveQualityAware;
  if (adaptive_constraint_mode && !enabled_) {
    throw std::invalid_argument(
            "adaptive roughness constraint modes require "
            "enable_roughness_innovation_gate=true");
  }

  adaptive_memory_config_.trigger_score =
    adaptive_memory_trigger_score < 0.0 ?
    roughness_score_gate_ : adaptive_memory_trigger_score;
  near_chain_quality_config_.yaw_disagreement_good_rad =
    near_chain_quality_yaw_good_deg * kDegreesToRadians;
  near_chain_quality_config_.yaw_disagreement_bad_rad =
    near_chain_quality_yaw_bad_deg * kDegreesToRadians;
  std::string adaptive_parameter_error;
  if (!validateAdaptiveRoughnessMemoryConfig(
      adaptive_memory_config_, &adaptive_parameter_error))
  {
    throw std::invalid_argument(
            "Invalid adaptive roughness memory parameters: " +
            adaptive_parameter_error);
  }
  adaptive_roughness_memory_.configure(adaptive_memory_config_);
  if (!validateAdaptiveConstraintPolicyConfig(
      adaptive_constraint_policy_config_, &adaptive_parameter_error))
  {
    throw std::invalid_argument(
            "Invalid adaptive constraint policy parameters: " +
            adaptive_parameter_error);
  }
  if (
    near_chain_quality_config_.calibrated &&
    !validateNearChainQualityConfig(
      near_chain_quality_config_, &adaptive_parameter_error))
  {
    throw std::invalid_argument(
            "Invalid near-chain quality parameters: " +
            adaptive_parameter_error);
  }
  if (
    !std::isfinite(adaptive_dynamic_threshold_max_) ||
    adaptive_dynamic_threshold_max_ < 0.0 ||
    adaptive_dynamic_threshold_max_ > 1.0)
  {
    throw std::invalid_argument(
            "roughness_adaptive_dynamic_threshold_max must be in [0, 1]");
  }
  rotation_consistency_config_.history_size = static_cast<std::size_t>(
    std::max(rotation_history_size, 0));
  rotation_consistency_config_.minimum_history_batches =
    static_cast<std::size_t>(
    std::max(rotation_minimum_history_batches, 0));
  rotation_consistency_config_.yaw_scale_floor_rad =
    rotation_yaw_scale_floor_deg * kDegreesToRadians;
  rotation_consistency_config_.maximum_repeat_state_gap =
    rotation_maximum_repeat_state_gap;
  rotation_consistency_config_.stable_recovery_batches =
    static_cast<std::size_t>(
    std::max(rotation_stable_recovery_batches, 0));
  if (!validateNearChainRotationConsistencyConfig(
      rotation_consistency_config_, &adaptive_parameter_error))
  {
    throw std::invalid_argument(
            "Invalid near-chain rotation consistency parameters: " +
            adaptive_parameter_error);
  }
  near_chain_rotation_consistency_.configure(rotation_consistency_config_);
  if (
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveMemoryOnly)
  {
    RCLCPP_WARN(
      get_logger(),
      "adaptive_memory_only is an ablation mode and must not be used as "
      "the production default.");
  }
  if (
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveQualityAware &&
    !near_chain_quality_config_.calibrated)
  {
    RCLCPP_WARN(
      get_logger(),
      "adaptive_quality_aware has no calibrated candidate-quality model; "
      "all candidate interventions will fail open while exact distributions "
      "are collected.");
  }
  if (!std::isfinite(near_chain_recovery_duration_s_)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_near_chain_recovery_duration_s; using 2.0.");
    near_chain_recovery_duration_s_ = 2.0;
  }
  near_chain_recovery_duration_s_ =
    std::max(near_chain_recovery_duration_s_, 0.0);
  if (!std::isfinite(near_chain_loop_closing_cooldown_s_)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_near_chain_loop_closing_cooldown_s; using 0.0.");
    near_chain_loop_closing_cooldown_s_ = 0.0;
  }
  near_chain_loop_closing_cooldown_s_ =
    std::max(near_chain_loop_closing_cooldown_s_, 0.0);
  near_chain_min_rough_samples_ = static_cast<std::size_t>(
    std::max(near_chain_min_rough_samples, 1));
  near_chain_observer_min_parity_candidates_ = static_cast<std::size_t>(
    std::max(near_chain_observer_min_parity_candidates, 1));
  if (!std::isfinite(near_chain_gated_min_response_)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_near_chain_gated_min_response; using 1.1.");
    near_chain_gated_min_response_ = 1.1;
  }
  near_chain_gated_min_response_ =
    std::clamp(near_chain_gated_min_response_, 0.0, 2.0);
  if (!std::isfinite(spatial_memory_radius_m_)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_spatial_memory_radius_m; using 0.6.");
    spatial_memory_radius_m_ = 0.6;
  }
  if (!std::isfinite(spatial_memory_min_revisit_separation_s_)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_spatial_memory_min_revisit_separation_s; using 20.0.");
    spatial_memory_min_revisit_separation_s_ = 20.0;
  }
  if (!std::isfinite(spatial_memory_min_heading_delta_deg)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_spatial_memory_min_heading_delta_deg; using 126.0.");
    spatial_memory_min_heading_delta_deg = 126.0;
  }
  if (!std::isfinite(spatial_memory_merge_radius_m_)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_spatial_memory_merge_radius_m; using 0.10.");
    spatial_memory_merge_radius_m_ = 0.10;
  }
  spatial_memory_radius_m_ = std::max(spatial_memory_radius_m_, 0.0);
  spatial_memory_min_revisit_separation_s_ =
    std::max(spatial_memory_min_revisit_separation_s_, 0.0);
  spatial_memory_min_heading_delta_rad_ =
    std::clamp(spatial_memory_min_heading_delta_deg, 0.0, 180.0) *
    kDegreesToRadians;
  spatial_memory_merge_radius_m_ = std::clamp(
    spatial_memory_merge_radius_m_, 0.0, spatial_memory_radius_m_);
  spatial_memory_max_zones_ = static_cast<std::size_t>(
    std::max(spatial_memory_max_zones, 1));
  if (!std::isfinite(spatial_passage_radius_m_)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_spatial_passage_radius_m; using 0.55.");
    spatial_passage_radius_m_ = 0.55;
  }
  if (!std::isfinite(spatial_passage_max_age_s_)) {
    RCLCPP_WARN(
      get_logger(),
      "Non-finite roughness_spatial_passage_max_age_s; using 3.5.");
    spatial_passage_max_age_s_ = 3.5;
  }
  spatial_passage_radius_m_ = std::max(spatial_passage_radius_m_, 0.0);
  spatial_passage_max_age_s_ = std::max(spatial_passage_max_age_s_, 0.0);
  const bool legacy_fixed_memory_mode =
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kLegacyFixedMemory;
  if (!legacy_fixed_memory_mode && enable_spatial_memory_gate_) {
    RCLCPP_WARN(
      get_logger(),
      "Fixed spatial passage memory is only active in "
      "legacy_fixed_memory; disabling it for mode '%s'.",
      roughness_constraint_mode_name_.c_str());
    enable_spatial_memory_gate_ = false;
    spatial_memory_use_pose_graph_anchors_ = false;
    enable_spatial_passage_hysteresis_ = false;
  }
  if (enable_spatial_memory_gate_ && !enable_near_chain_gate_) {
    RCLCPP_WARN(
      get_logger(),
      "roughness_enable_spatial_memory_gate requires "
      "roughness_enable_near_chain_gate; disabling spatial memory.");
    enable_spatial_memory_gate_ = false;
  }
  if (spatial_memory_use_pose_graph_anchors_ && !enable_spatial_memory_gate_) {
    RCLCPP_WARN(
      get_logger(),
      "roughness_spatial_memory_use_pose_graph_anchors requires "
      "roughness_enable_spatial_memory_gate; disabling pose-graph anchors.");
    spatial_memory_use_pose_graph_anchors_ = false;
  }
  if (
    enable_spatial_passage_hysteresis_ &&
    (!enable_near_chain_gate_ || !enable_spatial_memory_gate_))
  {
    RCLCPP_WARN(
      get_logger(),
      "roughness_enable_spatial_passage_hysteresis requires both "
      "roughness_enable_near_chain_gate and "
      "roughness_enable_spatial_memory_gate; disabling passage hysteresis.");
    enable_spatial_passage_hysteresis_ = false;
  }
  for (const auto & parameter : {
      rclcpp::Parameter(
        "roughness_enable_spatial_memory_gate",
        enable_spatial_memory_gate_),
      rclcpp::Parameter(
        "roughness_spatial_memory_use_pose_graph_anchors",
        spatial_memory_use_pose_graph_anchors_),
      rclcpp::Parameter(
        "roughness_enable_spatial_passage_hysteresis",
        enable_spatial_passage_hysteresis_),
      rclcpp::Parameter(
        "roughness_spatial_passage_radius_m",
        spatial_passage_radius_m_),
      rclcpp::Parameter(
        "roughness_spatial_passage_max_age_s",
        spatial_passage_max_age_s_)})
  {
    const auto result = set_parameter(parameter);
    if (!result.successful) {
      throw std::runtime_error(
              "Failed to publish effective roughness parameter " +
              parameter.get_name() + ": " + result.reason);
    }
  }
  post_confirmation_max_age_s_ =
    std::max(post_confirmation_max_age_s_, 0.0);
  post_confirmation_min_base_scans_ = static_cast<std::size_t>(
    std::max(post_confirmation_min_base_scans, 1));
  post_confirmation_max_base_scans_ = static_cast<std::size_t>(
    std::max(post_confirmation_max_base_scans, 1));
  post_confirmation_min_base_scans_ = std::min(
    post_confirmation_min_base_scans_,
    post_confirmation_max_base_scans_);
  post_confirmation_min_response_ =
    std::clamp(post_confirmation_min_response_, 0.0, 1.0);
  post_confirmation_min_response_gain_ =
    std::max(post_confirmation_min_response_gain_, 0.0);
  post_confirmation_min_covariance_improvement_ratio_ =
    std::max(post_confirmation_min_covariance_improvement_ratio_, 1.0);
  post_confirmation_translation_disagreement_gate_m_ =
    std::max(post_confirmation_translation_disagreement_gate_m_, 0.0);
  post_confirmation_yaw_disagreement_gate_rad_ =
    std::max(post_confirmation_yaw_disagreement_gate_deg, 0.0) *
    kDegreesToRadians;
  roughness_place_descriptor_bins_ = static_cast<std::size_t>(
    std::max(roughness_place_descriptor_bins, 2));
  roughness_place_max_observations_per_event_ = static_cast<std::size_t>(
    std::max(roughness_place_max_observations_per_event, 3));
  roughness_place_max_events_ = static_cast<std::size_t>(
    std::max(roughness_place_max_events, 1));
  if (roughness_place_candidate_top_k < 1) {
    throw std::invalid_argument("roughness_place_candidate_top_k must be >= 1");
  }
  if (roughness_place_min_state_separation < 1) {
    throw std::invalid_argument(
            "roughness_place_min_state_separation must be >= 1");
  }
  if (
    !std::isfinite(roughness_place_min_revisit_separation_s_) ||
    roughness_place_min_revisit_separation_s_ < 0.0)
  {
    throw std::invalid_argument(
            "roughness_place_min_revisit_separation_s must be finite and >= 0");
  }
  roughness_place_candidate_top_k_ = static_cast<std::size_t>(
    roughness_place_candidate_top_k);
  roughness_place_min_state_separation_ = static_cast<std::size_t>(
    roughness_place_min_state_separation);
  if (roughness_place_reference_chain_scan_count < 2) {
    throw std::invalid_argument(
            "roughness_place_reference_chain_scan_count must be >= 2");
  }
  roughness_place_scan_chain_config_.scan_count = static_cast<std::size_t>(
    roughness_place_reference_chain_scan_count);
  if (roughness_place_reference_chain_max_state_gap < 0) {
    throw std::invalid_argument(
            "roughness_place_reference_chain_max_state_gap must be >= 0");
  }
  roughness_place_scan_chain_config_.maximum_anchor_state_gap =
    static_cast<std::size_t>(roughness_place_reference_chain_max_state_gap);
  if (!validRoughnessPlaceScanChainConfig(roughness_place_scan_chain_config_)) {
    throw std::invalid_argument("Invalid roughness place reference chain parameter");
  }
  roughness_place_constraint_thresholds_.maximum_proposal_disagreement_yaw_rad =
    roughness_place_max_proposal_disagreement_yaw_deg * kDegreesToRadians;
  roughness_place_constraint_thresholds_.maximum_fine_yaw_covariance_rad2 =
    roughness_place_max_fine_yaw_covariance_deg2 *
    kDegreesToRadians * kDegreesToRadians;
  roughness_place_constraint_thresholds_.maximum_bidirectional_yaw_error_rad =
    roughness_place_max_bidirectional_yaw_error_deg * kDegreesToRadians;
  roughness_place_constraint_thresholds_.maximum_pair_alignment_yaw_error_rad =
    roughness_place_max_pair_alignment_yaw_error_deg * kDegreesToRadians;
  roughness_place_constraint_yaw_variance_floor_rad2_ =
    roughness_place_constraint_yaw_variance_floor_deg2 *
    kDegreesToRadians * kDegreesToRadians;
  if (!validRoughnessPlaceConstraintThresholds(
      roughness_place_constraint_thresholds_))
  {
    throw std::invalid_argument("Invalid roughness place quality threshold");
  }
  if (
    !std::isfinite(roughness_place_constraint_pair_covariance_scale_) ||
    roughness_place_constraint_pair_covariance_scale_ < 0.0 ||
    !std::isfinite(roughness_place_constraint_group_covariance_scale_) ||
    roughness_place_constraint_group_covariance_scale_ <= 0.0 ||
    !std::isfinite(roughness_place_constraint_xy_variance_floor_m2_) ||
    roughness_place_constraint_xy_variance_floor_m2_ < 0.0 ||
    !std::isfinite(roughness_place_constraint_yaw_variance_floor_rad2_) ||
    roughness_place_constraint_yaw_variance_floor_rad2_ < 0.0)
  {
    throw std::invalid_argument("Invalid roughness place constraint covariance parameter");
  }
  std::transform(
    roughness_place_constraint_mode_.begin(),
    roughness_place_constraint_mode_.end(),
    roughness_place_constraint_mode_.begin(),
    [](const unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
  if (
    roughness_place_constraint_mode_ != "observe" &&
    roughness_place_constraint_mode_ != "soft")
  {
    throw std::invalid_argument(
            "roughness_place_constraint_mode must be 'observe' or 'soft'");
  }
  std::transform(
    roughness_place_constraint_anchor_strategy_.begin(),
    roughness_place_constraint_anchor_strategy_.end(),
    roughness_place_constraint_anchor_strategy_.begin(),
    [](const unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
  if (
    roughness_place_constraint_anchor_strategy_ != "best_single" &&
    roughness_place_constraint_anchor_strategy_ != "validated_pair")
  {
    throw std::invalid_argument(
            "roughness_place_constraint_anchor_strategy must be "
            "'best_single' or 'validated_pair'");
  }
  if (
    enable_roughness_place_scan_match_observer_ &&
    !enable_roughness_place_recognition_observer_)
  {
    throw std::invalid_argument(
            "roughness_enable_place_scan_match_observer requires "
            "roughness_enable_place_recognition_observer=true");
  }
  if (
    roughness_place_evaluate_both_directions_ &&
    !enable_roughness_place_scan_match_observer_)
  {
    throw std::invalid_argument(
            "roughness_place_evaluate_both_directions=true requires "
            "roughness_enable_place_scan_match_observer=true");
  }
  if (
    roughness_place_enable_reference_chain_observer_ &&
    !enable_roughness_place_scan_match_observer_)
  {
    throw std::invalid_argument(
            "roughness_place_enable_reference_chain_observer=true requires "
            "roughness_enable_place_scan_match_observer=true");
  }
  if (
    roughness_place_constraint_mode_ == "soft" &&
    !enable_roughness_place_scan_match_observer_)
  {
    throw std::invalid_argument(
            "roughness_place_constraint_mode=soft requires "
            "roughness_enable_place_scan_match_observer=true");
  }
  if (
    roughness_place_enable_reference_chain_observer_ &&
    roughness_place_constraint_mode_ != "observe")
  {
    throw std::invalid_argument(
            "roughness_place_enable_reference_chain_observer=true currently "
            "requires roughness_place_constraint_mode=observe");
  }
  if (
    roughness_place_constraint_mode_ == "soft" &&
    roughness_place_constraint_anchor_strategy_ == "validated_pair")
  {
    throw std::invalid_argument(
            "roughness_place_constraint_anchor_strategy=validated_pair is "
            "observe-only until atomic graph insertion is available");
  }
  {
    std::lock_guard<std::mutex> place_lock(roughness_place_mutex_);
    roughness_place_event_tracker_.configure(
      roughness_place_descriptor_bins_,
      roughness_place_max_observations_per_event_);
    roughness_place_scan_chain_tracker_.configure(
      roughness_place_scan_chain_config_);
    roughness_place_events_.clear();
    roughness_place_scan_chain_events_.clear();
    roughness_place_injected_passages_.clear();
    roughness_place_reference_chain_observed_passages_.clear();
  }
  diagnostics_period_s_ = std::max(diagnostics_period_s_, 0.1);
  max_pending_scans_ = static_cast<std::size_t>(std::max(max_pending_scans, 1));

  auto normalize_mode = [](std::string & value) {
      std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char character) {
          return static_cast<char>(std::tolower(character));
        });
    };
  normalize_mode(scan_conditioning_mode_name_);
  normalize_mode(scan_conditioning_stabilize_fallback_mode_);
  normalize_mode(scan_conditioning_residual_fallback_mode_);
  normalize_mode(scan_conditioning_residual_direction_name_);
  if (scan_conditioning_mode_name_ == "observe") {
    scan_conditioning_mode_ = ScanConditioningMode::kObserve;
  } else if (scan_conditioning_mode_name_ == "drop") {
    scan_conditioning_mode_ = ScanConditioningMode::kDrop;
  } else if (scan_conditioning_mode_name_ == "throttle") {
    scan_conditioning_mode_ = ScanConditioningMode::kThrottle;
  } else if (scan_conditioning_mode_name_ == "quarantine") {
    scan_conditioning_mode_ = ScanConditioningMode::kQuarantine;
  } else if (scan_conditioning_mode_name_ == "stabilize") {
    scan_conditioning_mode_ = ScanConditioningMode::kStabilize;
  } else if (scan_conditioning_mode_name_ == "residual_filter") {
    scan_conditioning_mode_ = ScanConditioningMode::kResidualFilter;
  } else {
    throw std::invalid_argument(
            "Unsupported roughness_scan_conditioning_mode: " +
            scan_conditioning_mode_name_);
  }
  for (const std::string * fallback : {
      &scan_conditioning_stabilize_fallback_mode_,
      &scan_conditioning_residual_fallback_mode_})
  {
    if (*fallback != "drop" && *fallback != "observe") {
      throw std::invalid_argument(
              "Scan conditioning fallback must be 'drop' or 'observe': " +
              *fallback);
    }
  }
  const auto residual_direction = parseResidualFilterDirection(
    scan_conditioning_residual_direction_name_);
  if (!residual_direction) {
    throw std::invalid_argument(
            "Unsupported roughness_scan_conditioning_residual_direction: " +
            scan_conditioning_residual_direction_name_);
  }
  scan_conditioning_residual_direction_ = *residual_direction;

  auto require_nonnegative = [](const double value, const char * name) {
      if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
      }
    };
  auto require_ratio = [](const double value, const char * name) {
      if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string(name) + " must be in [0, 1]");
      }
    };
  for (const auto & value : {
      std::pair<double, const char *>{
        scan_conditioning_max_scan_imu_dt_s_, "conditioning max scan/IMU dt"},
      {scan_conditioning_hold_duration_s_, "conditioning hold duration"},
      {scan_conditioning_hard_hold_duration_s_, "conditioning hard hold duration"},
      {scan_conditioning_hard_score_gate_, "conditioning hard score gate"},
      {scan_conditioning_throttle_min_publish_period_s_, "conditioning throttle period"},
      {scan_conditioning_max_continuous_drop_duration_s_, "conditioning drop bridge duration"},
      {scan_conditioning_stabilize_max_source_age_s_, "stabilize source age"},
      {scan_conditioning_stabilize_max_event_duration_s_, "stabilize event duration"},
      {scan_conditioning_stabilize_min_publish_period_s_, "stabilize publish period"},
      {scan_conditioning_stabilize_max_translation_m_, "stabilize translation"},
      {scan_conditioning_stabilize_max_yaw_rad_, "stabilize yaw"},
      {scan_conditioning_stabilize_max_roll_pitch_rad_, "stabilize roll/pitch"},
      {scan_conditioning_residual_max_source_age_s_, "residual source age"},
      {scan_conditioning_residual_max_translation_m_, "residual translation"},
      {scan_conditioning_residual_max_yaw_rad_, "residual yaw"},
      {scan_conditioning_residual_max_roll_pitch_rad_, "residual roll/pitch"},
      {scan_conditioning_residual_threshold_m_, "residual threshold"}})
  {
    require_nonnegative(value.first, value.second);
  }
  for (const auto & value : {
      std::pair<double, const char *>{
        scan_conditioning_stabilize_min_cached_valid_ratio_,
        "stabilize cached valid ratio"},
      {scan_conditioning_stabilize_min_output_valid_ratio_,
        "stabilize output valid ratio"},
      {scan_conditioning_residual_min_cached_valid_ratio_,
        "residual cached valid ratio"},
      {scan_conditioning_residual_min_overlap_ratio_,
        "residual overlap ratio"},
      {scan_conditioning_residual_min_output_valid_ratio_,
        "residual output valid ratio"},
      {scan_conditioning_residual_max_removed_ratio_,
        "residual maximum removed ratio"},
      {scan_conditioning_quarantine_min_valid_ratio_,
        "quarantine minimum valid ratio"}})
  {
    require_ratio(value.first, value.second);
  }
  scan_conditioning_hard_score_gate_ = std::max(
    scan_conditioning_hard_score_gate_, roughness_score_gate_);
  scan_conditioning_stabilize_min_output_valid_beams_ =
    static_cast<std::size_t>(
    std::max(conditioning_stabilize_min_output_valid_beams, 1));
  scan_conditioning_residual_neighbor_bins_ = static_cast<std::size_t>(
    std::max(conditioning_residual_neighbor_bins, 0));
  scan_conditioning_residual_min_overlap_beams_ = static_cast<std::size_t>(
    std::max(conditioning_residual_min_overlap_beams, 1));
  scan_conditioning_residual_min_output_valid_beams_ = static_cast<std::size_t>(
    std::max(conditioning_residual_min_output_valid_beams, 1));
  scan_conditioning_recovery_config_.recovery_scan_count =
    static_cast<std::size_t>(
    std::max(conditioning_quarantine_recovery_scan_count, 1));
  validateRoughnessRecoveryConfig(scan_conditioning_recovery_config_);
  scan_conditioning_recovery_gate_.configure(scan_conditioning_recovery_config_);
  scan_conditioning_imu_fail_open_gate_.configure(
    scan_conditioning_recovery_config_.fault_after_s,
    std::max(scan_conditioning_recovery_config_.max_scan_gap_s, 1.0e-6));
  scan_conditioning_imu_requirements_ = imuQualityRequirements(
    roughness_thresholds_.tilt_deg,
    roughness_thresholds_.gyro_xy_rad_s,
    roughness_thresholds_.accel_norm_deviation_m_s2,
    roughness_thresholds_.tilt_rate_deg_s,
    roughness_thresholds_.accel_norm_rate_m_s3);
  if (enable_scan_conditioning_ && !enabled_) {
    RCLCPP_WARN(
      get_logger(),
      "roughness scan conditioning requires enable_roughness_innovation_gate; disabling it.");
    enable_scan_conditioning_ = false;
  }

  std::stringstream stamp_stream(experimental_reject_stamps);
  std::string stamp_token;
  while (std::getline(stamp_stream, stamp_token, ',')) {
    stamp_token.erase(
      std::remove_if(
        stamp_token.begin(),
        stamp_token.end(),
        [](const unsigned char value) {return std::isspace(value) != 0;}),
      stamp_token.end());
    if (stamp_token.empty()) {
      continue;
    }
    try {
      std::size_t consumed = 0;
      const std::int64_t stamp_ns = std::stoll(stamp_token, &consumed);
      if (consumed != stamp_token.size() || stamp_ns <= 0) {
        throw std::invalid_argument("invalid scan timestamp");
      }
      experimental_reject_scan_stamps_ns_.insert(stamp_ns);
    } catch (const std::exception &) {
      RCLCPP_WARN(
        get_logger(),
        "Ignoring invalid roughness experimental reject stamp '%s'.",
        stamp_token.c_str());
    }
  }
  if (!experimental_reject_scan_stamps_ns_.empty()) {
    RCLCPP_WARN(
      get_logger(),
      "EXPERIMENTAL BAG-ONLY scan rejection enabled for %zu exact timestamp(s).",
      experimental_reject_scan_stamps_ns_.size());
  }

  last_diagnostics_at_ = std::chrono::steady_clock::now();
  if (enabled_) {
    auto imu_qos = rclcpp::SensorDataQoS().keep_last(imu_qos_depth_);
    if (imu_qos_reliable_) {
      imu_qos.reliable();
    }
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      imu_qos,
      std::bind(&RoughnessAwareSlamToolbox::imuCallback, this, std::placeholders::_1));
    RCLCPP_INFO(
      get_logger(),
      "IMU subscription QoS: reliability=%s depth=%zu",
      imu_qos_reliable_ ? "reliable" : "best_effort",
      imu_qos_depth_);
    pending_timer_ = create_wall_timer(
      std::chrono::milliseconds(20),
      [this]() {
        flushPendingScans(true);
        logDiagnostics();
      });
  }

  RCLCPP_INFO(
    get_logger(),
    "Roughness innovation gate: enabled=%s mode=%s imu=%s "
    "window=[-%.3f,+%.3f]s score>=%.3f sustained>=%zu samples/%.3fs "
    "innovation>=%.3fm or %.3fdeg near_chain_gate=%s "
    "near_chain_samples=%zu recovery=%.3fs gated_response=%.3f "
    "suppress_loop_closing=%s loop_cooldown=%.3fs "
    "spatial_memory=%s pose_graph_anchors=%s radius=%.3fm "
    "revisit_separation=%.3fs heading_delta=%.3fdeg max_zones=%zu "
    "spatial_passage=%s passage_radius=%.3fm passage_max_age=%.3fs "
    "post_confirmation=%s "
    "base_scans=%zu..%zu max_age=%.3fs",
    enabled_ ? "true" : "false",
    mode_.c_str(),
    imu_topic_.c_str(),
    window_before_s_,
    window_after_s_,
    roughness_score_gate_,
    min_rough_samples_,
    min_rough_duration_s_,
    translation_innovation_gate_m_,
    yaw_innovation_gate_rad_ * kRadiansToDegrees,
    enable_near_chain_gate_ ? "true" : "false",
    near_chain_min_rough_samples_,
    near_chain_recovery_duration_s_,
    near_chain_gated_min_response_,
    disable_loop_closing_during_near_chain_gate_ ? "true" : "false",
    near_chain_loop_closing_cooldown_s_,
    enable_spatial_memory_gate_ ? "true" : "false",
    spatial_memory_use_pose_graph_anchors_ ? "true" : "false",
    spatial_memory_radius_m_,
    spatial_memory_min_revisit_separation_s_,
    spatial_memory_min_heading_delta_rad_ * kRadiansToDegrees,
    spatial_memory_max_zones_,
    enable_spatial_passage_hysteresis_ ? "true" : "false",
    spatial_passage_radius_m_,
    spatial_passage_max_age_s_,
    enable_post_confirmation_ ? "true" : "false",
    post_confirmation_min_base_scans_,
    post_confirmation_max_base_scans_,
    post_confirmation_max_age_s_);
  RCLCPP_INFO(
    get_logger(),
    "Adaptive near-chain policy: mode=%s exact_observer=%s "
    "memory_score=%.3f..%.3f recovery_score=%.3f "
    "duration=%.3f..%.3fs distance=%.3f..%.3fm recovery_tau=%.3fs "
    "quality_calibrated=%s risk=%.3f/%.3f dynamic_threshold_max=%.3f "
    "soft_weighting_available=false fail_open=true",
    roughnessConstraintModeName(roughness_constraint_mode_),
    (enable_near_chain_candidate_observer_ ||
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveMemoryObserve ||
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveMemoryOnly ||
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveQualityAware) ? "true" : "false",
    adaptive_memory_config_.trigger_score,
    adaptive_memory_config_.saturation_score,
    adaptive_memory_config_.recovery_score,
    adaptive_memory_config_.minimum_duration_s,
    adaptive_memory_config_.maximum_duration_s,
    adaptive_memory_config_.minimum_distance_m,
    adaptive_memory_config_.maximum_distance_m,
    adaptive_memory_config_.stable_recovery_time_constant_s,
    near_chain_quality_config_.calibrated ? "true" : "false",
    adaptive_constraint_policy_config_.downweight_risk,
    adaptive_constraint_policy_config_.reject_risk,
    adaptive_dynamic_threshold_max_);
  RCLCPP_INFO(
    get_logger(),
    "SLAM-owned roughness scan conditioning: enabled=%s mode=%s "
    "nearest_imu=%s max_scan_imu_dt=%.3fs score_gate=%.3f hard_gate=%.3f "
    "hold=%.3fs hard_hold=%.3fs quarantine_recovery=%zu/%.3fs "
    "quarantine_gap=%.3fs imu_fail_open=%s residual_direction=%s "
    "residual_threshold=%.3fm residual_cap=%.3f",
    enable_scan_conditioning_ ? "true" : "false",
    scanConditioningModeName(scan_conditioning_mode_),
    scan_conditioning_use_nearest_imu_ ? "true" : "false",
    scan_conditioning_max_scan_imu_dt_s_,
    roughness_score_gate_,
    scan_conditioning_hard_score_gate_,
    scan_conditioning_hold_duration_s_,
    scan_conditioning_hard_hold_duration_s_,
    scan_conditioning_recovery_config_.recovery_scan_count,
    scan_conditioning_recovery_config_.min_stable_duration_s,
    scan_conditioning_recovery_config_.max_scan_gap_s,
    scan_conditioning_quarantine_fail_open_on_imu_fault_ ? "true" : "false",
    scan_conditioning_residual_direction_name_.c_str(),
    scan_conditioning_residual_threshold_m_,
    scan_conditioning_residual_max_removed_ratio_);
  RCLCPP_INFO(
    get_logger(),
    "Roughness place recognition: enabled=%s scan_match=%s both_directions=%s "
    "reference_chain=%s chain_scans=%zu chain_max_distance=%.3fm mode=%s "
    "chain_max_state_gap=%zu chain_max_time_gap=%.3fs "
    "anchor_strategy=%s "
    "descriptor_bins=%zu max_observations_per_event=%zu max_events=%zu "
    "candidate_top_k=%zu min_revisit=%.3fs min_state_separation=%zu "
    "fine_response=%.3f proposal=%.3fm/%.3fdeg "
    "bidirectional=%.3fm/%.3fdeg pair=%.3fm/%.3fdeg "
    "distance_axis=odometry direction_invariant=true fail_open=true",
    enable_roughness_place_recognition_observer_ ? "true" : "false",
    enable_roughness_place_scan_match_observer_ ? "true" : "false",
    roughness_place_evaluate_both_directions_ ? "true" : "false",
    roughness_place_enable_reference_chain_observer_ ? "true" : "false",
    roughness_place_scan_chain_config_.scan_count,
    roughness_place_scan_chain_config_.maximum_anchor_distance_m,
    roughness_place_constraint_mode_.c_str(),
    roughness_place_scan_chain_config_.maximum_anchor_state_gap,
    roughness_place_scan_chain_config_.maximum_anchor_time_gap_s,
    roughness_place_constraint_anchor_strategy_.c_str(),
    roughness_place_descriptor_bins_,
    roughness_place_max_observations_per_event_,
    roughness_place_max_events_,
    roughness_place_candidate_top_k_,
    roughness_place_min_revisit_separation_s_,
    roughness_place_min_state_separation_,
    roughness_place_constraint_thresholds_.minimum_fine_response,
    roughness_place_constraint_thresholds_.
    maximum_proposal_disagreement_translation_m,
    roughness_place_constraint_thresholds_.maximum_proposal_disagreement_yaw_rad *
    kRadiansToDegrees,
    roughness_place_constraint_thresholds_.
    maximum_bidirectional_translation_error_m,
    roughness_place_constraint_thresholds_.maximum_bidirectional_yaw_error_rad *
    kRadiansToDegrees,
    roughness_place_constraint_thresholds_.
    maximum_pair_alignment_translation_error_m,
    roughness_place_constraint_thresholds_.maximum_pair_alignment_yaw_error_rad *
    kRadiansToDegrees);

  clear_queue_service_ = create_service<slam_toolbox::srv::ClearQueue>(
    "slam_toolbox/clear_queue",
    std::bind(
      &RoughnessAwareSlamToolbox::clearQueueCallback,
      this,
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3));
  finalize_pending_service_ = create_service<slam_toolbox::srv::ClearQueue>(
    "slam_toolbox/finalize_roughness_pending",
    std::bind(
      &RoughnessAwareSlamToolbox::finalizePendingScansCallback,
      this,
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3));
  stop_scan_worker_.store(false);
  scan_worker_ = std::thread(&RoughnessAwareSlamToolbox::runScanWorker, this);
}

bool RoughnessAwareSlamToolbox::deserializePoseGraphCallback(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<slam_toolbox::srv::DeserializePoseGraph::Request> req,
  std::shared_ptr<slam_toolbox::srv::DeserializePoseGraph::Response> resp)
{
  if (req->match_type == toolbox_types::procType::LOCALIZE_AT_POSE) {
    RCLCPP_ERROR(
      get_logger(),
      "Requested a localization deserialization in non-localization mode.");
    return false;
  }

  std::lock_guard<std::mutex> frontend_lock(frontend_queue_mutex_);
  std::lock_guard<std::mutex> epoch_lock(mapper_epoch_mutex_);
  karto::Mapper * mapper_before = nullptr;
  {
    boost::mutex::scoped_lock mapper_lock(smapper_mutex_);
    mapper_before = smapper_ ? smapper_->getMapper() : nullptr;
  }
  const bool result =
    slam_toolbox::SlamToolbox::deserializePoseGraphCallback(
    request_header, req, resp);
  karto::Mapper * mapper_after = nullptr;
  {
    boost::mutex::scoped_lock mapper_lock(smapper_mutex_);
    mapper_after = smapper_ ? smapper_->getMapper() : nullptr;
  }
  if (mapper_after != mapper_before) {
    const std::uint64_t generation = mapper_generation_.fetch_add(1) + 1;
    std::size_t pending_count = 0;
    std::size_t queued_count = 0;
    {
      std::lock_guard<std::mutex> lock(roughness_mutex_);
      pending_count = pending_scans_.size();
      pending_scans_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(scan_queue_mutex_);
      queued_count = scan_queue_.size();
      scan_queue_.clear();
    }
    scan_queue_cv_.notify_all();
    resetMapperDerivedState("mapper_deserialize_complete");
    RCLCPP_INFO(
      get_logger(),
      "Pose graph mapper epoch advanced to %llu; dropped %zu pending and "
      "%zu queued scan(s).",
      static_cast<unsigned long long>(generation),
      pending_count,
      queued_count);
  } else {
    RCLCPP_WARN(
      get_logger(),
      "Pose graph deserialize did not replace the mapper; preserving mapper "
      "epoch and roughness state.");
  }
  return result;
}

bool RoughnessAwareSlamToolbox::clearQueueCallback(
  const std::shared_ptr<rmw_request_id_t>/*request_header*/,
  const std::shared_ptr<slam_toolbox::srv::ClearQueue::Request>/*req*/,
  std::shared_ptr<slam_toolbox::srv::ClearQueue::Response> resp)
{
  std::lock_guard<std::mutex> frontend_lock(frontend_queue_mutex_);
  std::lock_guard<std::mutex> epoch_lock(mapper_epoch_mutex_);
  const std::uint64_t generation = mapper_generation_.fetch_add(1) + 1;

  std::size_t pending_count = 0;
  {
    std::lock_guard<std::mutex> lock(roughness_mutex_);
    pending_count = pending_scans_.size();
    pending_scans_.clear();
  }

  std::size_t queued_count = 0;
  {
    std::lock_guard<std::mutex> lock(scan_queue_mutex_);
    queued_count = scan_queue_.size();
    scan_queue_.clear();
  }
  scan_queue_cv_.notify_all();
  resetMapperDerivedState("queue_cleared");

  resp->status = true;
  RCLCPP_INFO(
    get_logger(),
    "Cleared %zu pending and %zu queued roughness-aware scan(s); "
    "mapper epoch is now %llu.",
    pending_count,
    queued_count,
    static_cast<unsigned long long>(generation));
  return true;
}

bool RoughnessAwareSlamToolbox::finalizePendingScansCallback(
  const std::shared_ptr<rmw_request_id_t>/*request_header*/,
  const std::shared_ptr<slam_toolbox::srv::ClearQueue::Request>/*req*/,
  std::shared_ptr<slam_toolbox::srv::ClearQueue::Response> resp)
{
  std::vector<std::pair<PendingScan, WindowDecision>> finalized;
  std::size_t complete_count = 0;
  {
    std::lock_guard<std::mutex> lock(roughness_mutex_);
    finalized.reserve(pending_scans_.size());
    for (const PendingScan & pending : pending_scans_) {
      const auto decision = windowDecisionLocked(
        messageStampNs(pending.scan->header.stamp));
      complete_count += decision.complete ? 1U : 0U;
      finalized.emplace_back(pending, decision);
    }
    pending_scans_.clear();
  }
  for (const auto & item : finalized) {
    queueAcceptedScan(item.first, item.second);
  }
  resp->status = true;
  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_PENDING_FINALIZE total=%zu complete=%zu fail_open=%zu",
    finalized.size(),
    complete_count,
    finalized.size() - complete_count);
  return true;
}

void RoughnessAwareSlamToolbox::laserCallback(
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
{
  if (enabled_) {
    const std::int64_t stamp_ns = messageStampNs(scan->header.stamp);
    received_scan_stamp_audit_.record(stamp_ns);
    ++received_scans_;
    if (experimental_reject_scan_stamps_ns_.count(stamp_ns) != 0U) {
      ++rejected_scans_;
      RCLCPP_WARN(
        get_logger(),
        "ROUGHNESS_EXPERIMENTAL_EXACT_REJECT stamp_ns=%lld action=reject",
        static_cast<long long>(stamp_ns));
      return;
    }
  }

  karto::Pose2 pose;
  if (!pose_helper_->getOdomPose(pose, scan->header.stamp)) {
    RCLCPP_WARN(get_logger(), "Failed to compute odom pose");
    return;
  }

  std::unique_lock<std::mutex> frontend_lock(frontend_queue_mutex_);
  std::uint64_t mapper_generation = 0;
  {
    std::lock_guard<std::mutex> epoch_lock(mapper_epoch_mutex_);
    karto::LaserRangeFinder * laser = getLaser(scan);
    if (!laser) {
      RCLCPP_WARN(
        get_logger(),
        "RoughnessAwareSlamToolbox: failed to create laser device for %s; discarding scan",
        scan->header.frame_id.c_str());
      return;
    }
    if (!shouldProcessScan(scan, pose)) {
      return;
    }
    mapper_generation = mapper_generation_.load();
  }

  if (!enabled_) {
    WindowDecision decision;
    decision.mapper_generation = mapper_generation;
    {
      std::lock_guard<std::mutex> lock(scan_queue_mutex_);
      scan_queue_.push_back(
        {scan, pose, decision, next_scan_sequence_++});
    }
    scan_queue_cv_.notify_one();
    ++queued_scans_;
    return;
  }

  std::optional<PendingScan> overflow;
  {
    std::lock_guard<std::mutex> lock(roughness_mutex_);
    if (pending_scans_.size() >= max_pending_scans_) {
      overflow = pending_scans_.front();
      pending_scans_.pop_front();
      ++pending_overflow_fail_open_;
      RCLCPP_WARN(
        get_logger(),
        "Roughness pending scan queue reached %zu; oldest scan is failing open.",
        max_pending_scans_);
    }
    pending_scans_.push_back(
      {
        scan,
        pose,
        std::chrono::steady_clock::now(),
        mapper_generation,
      });
  }
  frontend_lock.unlock();
  if (overflow) {
    queueAcceptedScan(*overflow, WindowDecision{});
  }
  flushPendingScans(false);
}

void RoughnessAwareSlamToolbox::imuCallback(
  sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  const std::int64_t stamp_ns = messageStampNs(msg->header.stamp);
  received_imu_stamp_audit_.record(stamp_ns);
  ++received_imus_;
  if (enable_scan_conditioning_) {
    const ImuQualityResult quality = validateImuQuality(
      *msg, scan_conditioning_imu_requirements_);
    if (!quality.valid) {
      ++scan_conditioning_imu_quality_reject_count_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "ROUGHNESS_SCAN_CONDITION_IMU_REJECT reason=%s stamp_ns=%lld",
        imuQualityReasonName(quality.reason),
        static_cast<long long>(stamp_ns));
      return;
    }
  }
  if (
    stamp_ns <= 0 ||
    !finiteVector(msg->angular_velocity) ||
    !finiteVector(msg->linear_acceleration))
  {
    return;
  }

  const double tilt_deg = tiltDeg(*msg);
  if (!std::isfinite(tilt_deg)) {
    return;
  }
  const double gyro_xy =
    std::hypot(msg->angular_velocity.x, msg->angular_velocity.y);
  const double accel_norm = std::sqrt(
    msg->linear_acceleration.x * msg->linear_acceleration.x +
    msg->linear_acceleration.y * msg->linear_acceleration.y +
    msg->linear_acceleration.z * msg->linear_acceleration.z);
  const double accel_deviation = std::fabs(accel_norm - kGravityMps2);

  std::vector<PendingScan> epoch_reset_pending;
  bool epoch_reset = false;
  {
    std::lock_guard<std::mutex> lock(roughness_mutex_);
    if (!imu_samples_.empty() && stamp_ns < imu_samples_.back().stamp_ns) {
      const double rollback_s =
        static_cast<double>(imu_samples_.back().stamp_ns - stamp_ns) * 1.0e-9;
      if (rollback_s < imu_epoch_reset_threshold_s_) {
        ++out_of_order_imu_samples_;
        return;
      }
      epoch_reset_pending.assign(pending_scans_.begin(), pending_scans_.end());
      pending_scans_.clear();
      imu_samples_.clear();
      previous_distinct_imu_.reset();
      ++imu_epoch_resets_;
      epoch_reset = true;
    }

    const bool duplicate =
      !imu_samples_.empty() && stamp_ns == imu_samples_.back().stamp_ns;
    const ImuSample * derivative_reference = nullptr;
    if (duplicate) {
      derivative_reference = previous_distinct_imu_ ?
        &*previous_distinct_imu_ : nullptr;
    } else if (!imu_samples_.empty()) {
      derivative_reference = &imu_samples_.back();
    }

    RoughnessSignal signal;
    signal.tilt_deg = tilt_deg;
    signal.gyro_xy_rad_s = gyro_xy;
    signal.accel_norm_deviation_m_s2 = accel_deviation;
    signal.tilt_rate_deg_s = std::numeric_limits<double>::quiet_NaN();
    signal.accel_norm_rate_m_s3 = std::numeric_limits<double>::quiet_NaN();
    if (derivative_reference) {
      const double dt_s =
        static_cast<double>(stamp_ns - derivative_reference->stamp_ns) * 1.0e-9;
      if (dt_s > 1.0e-6) {
        signal.tilt_rate_deg_s =
          std::fabs(tilt_deg - derivative_reference->signal.tilt_deg) / dt_s;
        signal.accel_norm_rate_m_s3 =
          std::fabs(
          accel_deviation -
          derivative_reference->signal.accel_norm_deviation_m_s2) / dt_s;
      }
    }

    ImuSample sample;
    sample.stamp_ns = stamp_ns;
    sample.signal = signal;
    sample.score = normalizedRoughnessScore(signal, roughness_thresholds_);
    sample.quality_valid = true;
    if (duplicate) {
      imu_samples_.back() = sample;
      ++duplicate_imu_samples_;
    } else {
      if (!imu_samples_.empty()) {
        previous_distinct_imu_ = imu_samples_.back();
      }
      imu_samples_.push_back(sample);
    }

    const std::int64_t buffer_duration_ns =
      static_cast<std::int64_t>(imu_buffer_duration_s_ * 1.0e9);
    while (
      imu_samples_.size() > 1 &&
      stamp_ns - imu_samples_.front().stamp_ns > buffer_duration_ns)
    {
      imu_samples_.pop_front();
    }
  }

  if (epoch_reset) {
    clearPostConfirmation("imu_epoch_reset");
    resetScanConditioningState("imu_epoch_reset");
  }
  if (!epoch_reset_pending.empty()) {
    RCLCPP_WARN(
      get_logger(),
      "IMU timestamp epoch reset; %zu pending scan(s) failed open.",
      epoch_reset_pending.size());
    for (const PendingScan & pending : epoch_reset_pending) {
      queueAcceptedScan(pending, WindowDecision{});
    }
  }
  flushPendingScans(false);
}

void RoughnessAwareSlamToolbox::flushPendingScans(const bool allow_wall_timeout)
{
  std::vector<std::pair<PendingScan, WindowDecision>> ready;
  const auto now = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(roughness_mutex_);
    while (!pending_scans_.empty()) {
      const PendingScan & pending = pending_scans_.front();
      const std::int64_t scan_stamp_ns = messageStampNs(pending.scan->header.stamp);
      const std::int64_t required_imu_stamp_ns =
        scan_stamp_ns + static_cast<std::int64_t>(window_after_s_ * 1.0e9);
      const bool imu_ready =
        !imu_samples_.empty() &&
        imu_samples_.back().stamp_ns > required_imu_stamp_ns;
      const bool timed_out =
        allow_wall_timeout &&
        std::chrono::duration<double>(now - pending.enqueued_at).count() >=
        pending_timeout_s_;
      if (!imu_ready && !timed_out) {
        break;
      }

      ready.emplace_back(
        pending,
        windowDecisionLocked(scan_stamp_ns));
      pending_scans_.pop_front();
    }
  }

  for (const auto & item : ready) {
    queueAcceptedScan(item.first, item.second);
  }
}

void RoughnessAwareSlamToolbox::queueAcceptedScan(
  const PendingScan & pending,
  const WindowDecision & decision)
{
  std::lock_guard<std::mutex> frontend_lock(frontend_queue_mutex_);
  WindowDecision queued_decision = decision;
  queued_decision.mapper_generation = pending.mapper_generation;
  {
    std::lock_guard<std::mutex> lock(scan_queue_mutex_);
    scan_queue_.push_back(
      {
        pending.scan,
        pending.odom_pose,
        queued_decision,
        next_scan_sequence_++,
      });
  }
  scan_queue_cv_.notify_one();
  ++queued_scans_;
  if (decision.complete) {
    ++complete_windows_;
  }
}

void RoughnessAwareSlamToolbox::runScanWorker()
{
  while (!stop_scan_worker_.load()) {
    if (isPaused(toolbox_types::PROCESSING)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    std::optional<QueuedScan> queued;
    std::size_t remaining = 0;
    {
      std::unique_lock<std::mutex> lock(scan_queue_mutex_);
      scan_queue_cv_.wait_for(
        lock,
        std::chrono::milliseconds(10),
        [this]() {
          return stop_scan_worker_.load() || !scan_queue_.empty();
        });
      if (stop_scan_worker_.load()) {
        return;
      }
      if (scan_queue_.empty()) {
        continue;
      }
      queued = std::move(scan_queue_.front());
      scan_queue_.pop_front();
      remaining = scan_queue_.size();
      scan_worker_in_flight_.fetch_add(1U);
    }
    if (remaining > 10) {
      RCLCPP_WARN(
        get_logger(),
        "Queue size has grown to: %zu. Recommend stopping until message is gone "
        "if online mapping.",
        remaining);
    }

    try {
      (void)processQueuedScan(*queued);
    } catch (const karto::Exception & ex) {
      RCLCPP_ERROR(
        get_logger(),
        "Karto exception while processing queued scan %llu: %s",
        static_cast<unsigned long long>(queued->sequence),
        ex.GetErrorMessage().c_str());
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(
        get_logger(),
        "Exception while processing queued scan %llu: %s",
        static_cast<unsigned long long>(queued->sequence),
        ex.what());
    } catch (...) {
      RCLCPP_ERROR(
        get_logger(),
        "Unknown exception while processing queued scan %llu.",
        static_cast<unsigned long long>(queued->sequence));
    }
    scan_worker_completed_.fetch_add(1U);
    scan_worker_in_flight_.fetch_sub(1U);
  }
}

karto::LocalizedRangeScan * RoughnessAwareSlamToolbox::processQueuedScan(
  const QueuedScan & queued)
{
  if (!queued.scan) {
    return nullptr;
  }

  std::lock_guard<std::mutex> epoch_lock(mapper_epoch_mutex_);
  const std::uint64_t current_generation = mapper_generation_.load();
  if (queued.decision.mapper_generation != current_generation) {
    ++stale_mapper_generation_scan_drop_count_;
    RCLCPP_WARN(
      get_logger(),
      "Dropping queued scan %llu from mapper generation %llu; "
      "current generation is %llu.",
      static_cast<unsigned long long>(queued.sequence),
      static_cast<unsigned long long>(queued.decision.mapper_generation),
      static_cast<unsigned long long>(current_generation));
    return nullptr;
  }

  karto::LaserRangeFinder * laser = getLaser(queued.scan);
  if (!laser) {
    RCLCPP_WARN(
      get_logger(),
      "RoughnessAwareSlamToolbox: failed to load laser device for queued scan %llu.",
      static_cast<unsigned long long>(queued.sequence));
    return nullptr;
  }

  karto::Pose2 odom_pose = queued.odom_pose;
  const std::optional<WindowDecision> window =
    enabled_ ? std::make_optional(queued.decision) : std::nullopt;
  return processScanWithWindow(laser, queued.scan, odom_pose, window);
}

RoughnessAwareSlamToolbox::WindowDecision
RoughnessAwareSlamToolbox::windowDecisionLocked(
  const std::int64_t scan_stamp_ns) const
{
  WindowDecision decision;
  if (imu_samples_.empty() || scan_stamp_ns <= 0) {
    return decision;
  }

  const std::int64_t lower_ns =
    scan_stamp_ns - static_cast<std::int64_t>(window_before_s_ * 1.0e9);
  const std::int64_t upper_ns =
    scan_stamp_ns + static_cast<std::int64_t>(window_after_s_ * 1.0e9);
  decision.complete =
    imu_samples_.front().stamp_ns <= lower_ns &&
    imu_samples_.back().stamp_ns > upper_ns;
  std::size_t current_rough_samples = 0;
  std::int64_t current_rough_start_ns = 0;
  auto update_peak = [](double & peak, const double value) {
      if (!std::isfinite(value)) {
        return;
      }
      peak = std::isfinite(peak) ? std::max(peak, value) : value;
    };
  for (const auto & sample : imu_samples_) {
    if (sample.stamp_ns < lower_ns) {
      continue;
    }
    if (sample.stamp_ns > upper_ns) {
      break;
    }
    update_peak(decision.peak_signal.tilt_deg, sample.signal.tilt_deg);
    update_peak(
      decision.peak_signal.gyro_xy_rad_s,
      sample.signal.gyro_xy_rad_s);
    update_peak(
      decision.peak_signal.accel_norm_deviation_m_s2,
      sample.signal.accel_norm_deviation_m_s2);
    update_peak(
      decision.peak_signal.tilt_rate_deg_s,
      sample.signal.tilt_rate_deg_s);
    update_peak(
      decision.peak_signal.accel_norm_rate_m_s3,
      sample.signal.accel_norm_rate_m_s3);
    if (std::isfinite(sample.score)) {
      decision.score =
        std::isfinite(decision.score) ?
        std::max(decision.score, sample.score) : sample.score;
      ++decision.sample_count;
      if (sample.score >= roughness_score_gate_) {
        if (current_rough_samples == 0) {
          current_rough_start_ns = sample.stamp_ns;
        }
        ++current_rough_samples;
        const double current_duration_s =
          static_cast<double>(sample.stamp_ns - current_rough_start_ns) * 1.0e-9;
        decision.rough_sample_count =
          std::max(decision.rough_sample_count, current_rough_samples);
        decision.rough_duration_s =
          std::max(decision.rough_duration_s, current_duration_s);
        if (
          current_rough_samples >= min_rough_samples_ &&
          current_duration_s >= min_rough_duration_s_)
        {
          decision.sustained = true;
        }
      } else {
        current_rough_samples = 0;
        current_rough_start_ns = 0;
      }
    }
  }
  long double nearest_dt_ns = std::numeric_limits<long double>::infinity();
  for (const auto & sample : imu_samples_) {
    const long double dt_ns = std::fabs(
      static_cast<long double>(sample.stamp_ns) -
      static_cast<long double>(scan_stamp_ns));
    if (dt_ns >= nearest_dt_ns) {
      continue;
    }
    nearest_dt_ns = dt_ns;
    decision.nearest_imu_available = true;
    decision.nearest_imu_quality_valid = sample.quality_valid;
    decision.nearest_imu_dt_s = static_cast<double>(dt_ns * 1.0e-9L);
    decision.nearest_imu_score = sample.score;
  }
  return decision;
}

karto::LocalizedRangeScan * RoughnessAwareSlamToolbox::addScan(
  karto::LaserRangeFinder * laser,
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
  karto::Pose2 & odom_pose)
{
  std::lock_guard<std::mutex> epoch_lock(mapper_epoch_mutex_);
  return processScanWithWindow(laser, scan, odom_pose, std::nullopt);
}

void RoughnessAwareSlamToolbox::updateAdmittedScanHeader(
  const std_msgs::msg::Header & header)
{
  // Match slam_toolbox's mapper -> transform lock order. updateMap() reads
  // under smapper_mutex_, while the transform publisher reads under
  // map_to_odom_mutex_.
  boost::mutex::scoped_lock mapper_lock(smapper_mutex_);
  boost::mutex::scoped_lock transform_lock(map_to_odom_mutex_);
  scan_header = header;
}

karto::LocalizedRangeScan * RoughnessAwareSlamToolbox::processScanWithWindow(
  karto::LaserRangeFinder * laser,
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
  karto::Pose2 & odom_pose,
  const std::optional<WindowDecision> & window)
{
  if (!enabled_) {
    updateAdmittedScanHeader(scan->header);
    return slam_toolbox::SlamToolbox::addScan(laser, scan, odom_pose);
  }

  sensor_msgs::msg::LaserScan::ConstSharedPtr effective_scan = scan;
  if (enable_scan_conditioning_) {
    const ScanConditioningResult conditioning = conditionScan(scan, window);
    if (conditioning.drop || !conditioning.scan) {
      return nullptr;
    }
    effective_scan = conditioning.scan;
  }

  auto process_effective_scan = [this, laser, &effective_scan, &odom_pose, &window]() {
      updateAdmittedScanHeader(effective_scan->header);
      return processAndTrackScan(laser, effective_scan, odom_pose, window);
    };

  const std::int64_t stamp_ns = messageStampNs(effective_scan->header.stamp);
  evaluatePendingPostConfirmation(laser, effective_scan, odom_pose, window);
  if (
    !window ||
    !window->complete ||
    !window->sustained ||
    !std::isfinite(window->score) ||
    window->score < roughness_score_gate_)
  {
    return process_effective_scan();
  }
  ++rough_scans_;

  FrozenScanContext frozen_context;
  const ScanMatchInnovation innovation =
    previewScanMatchInnovation(
    laser,
    effective_scan,
    odom_pose,
    nullptr,
    enable_post_confirmation_ ? &frozen_context : nullptr);
  if (!innovation.available) {
    ++unavailable_innovations_;
    return process_effective_scan();
  }

  const RoughnessInnovationDecision gate = roughnessInnovationDecision(
    window->score,
    window->complete,
    window->sustained,
    roughness_score_gate_,
    innovation.translation_m,
    innovation.yaw_rad,
    translation_innovation_gate_m_,
    yaw_innovation_gate_rad_);
  if (gate.reject) {
    ++innovation_candidates_;
  }
  bool post_confirmation_armed = false;
  if (gate.reject && enable_post_confirmation_) {
    post_confirmation_armed = armPostConfirmation(
      stamp_ns,
      window->score,
      gate,
      innovation,
      std::move(frozen_context));
  }
  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_INNOVATION_DIAG stamp_ns=%lld mode=%s imu_samples=%zu "
    "score=%.6f response=%.6f translation_m=%.6f yaw_deg=%.6f candidate=%s action=allow "
    "rough_samples=%zu rough_duration_s=%.6f sustained=%s "
    "dx_m=%.6f dy_m=%.6f covariance_xx=%.9g covariance_xy=%.9g "
    "covariance_yy=%.9g covariance_yawyaw=%.9g "
    "covariance_xy_max_eigenvalue=%.9g covariance_xy_condition=%.9g "
    "scan_match_covariance_weighted_disagreement_sq=%.9g "
    "odom_correction_x_m=%.6f odom_correction_y_m=%.6f "
    "odom_correction_yaw_deg=%.6f post_confirmation_armed=%s",
    static_cast<long long>(stamp_ns),
    mode_.c_str(),
    window->sample_count,
    window->score,
    innovation.response,
    innovation.translation_m,
    innovation.yaw_rad * kRadiansToDegrees,
    gate.reject ? "true" : "false",
    window->rough_sample_count,
    window->rough_duration_s,
    window->sustained ? "true" : "false",
    innovation.dx_m,
    innovation.dy_m,
    innovation.covariance_xx,
    innovation.covariance_xy,
    innovation.covariance_yy,
    innovation.covariance_yawyaw,
    innovation.covariance_xy_max_eigenvalue,
    innovation.covariance_xy_condition,
    innovation.scan_match_covariance_weighted_disagreement_sq,
    innovation.odom_correction_x_m,
    innovation.odom_correction_y_m,
    innovation.odom_correction_yaw_rad * kRadiansToDegrees,
    post_confirmation_armed ? "true" : "false");

  return process_effective_scan();
}

karto::LocalizedRangeScan * RoughnessAwareSlamToolbox::processAndTrackScan(
  karto::LaserRangeFinder * laser,
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
  karto::Pose2 & odom_pose,
  const std::optional<WindowDecision> & window)
{
  std::unique_ptr<karto::LocalizedRangeScan> range_scan(
    getLocalizedRangeScan(laser, scan, odom_pose));
  if (!range_scan) {
    return nullptr;
  }

  boost::mutex::scoped_lock lock(smapper_mutex_);
  karto::Mapper * mapper = smapper_->getMapper();
  if (!mapper) {
    RCLCPP_ERROR(get_logger(), "Roughness SLAM mapper is unavailable.");
    return nullptr;
  }

  const std::int64_t stamp_ns = messageStampNs(scan->header.stamp);
  const bool legacy_fixed_memory_mode =
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kLegacyFixedMemory;
  const bool adaptive_mode =
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveMemoryObserve ||
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveMemoryOnly ||
    roughness_constraint_mode_ ==
    RoughnessConstraintMode::kAdaptiveQualityAware;
  const bool exact_candidate_observer_requested =
    enable_near_chain_candidate_observer_ || adaptive_mode ||
    rotation_consistency_config_.enabled;
  const bool roughness_memory_observer_requested =
    exact_candidate_observer_requested ||
    enable_loop_closure_candidate_observer_;
  RoughnessNearChainGateInput near_chain_input;
  near_chain_input.enabled =
    enable_near_chain_gate_ && legacy_fixed_memory_mode;
  near_chain_input.scan_stamp_ns = stamp_ns;
  near_chain_input.imu_window_complete = window && window->complete;
  near_chain_input.rough_sample_count =
    window ? window->rough_sample_count : 0U;
  near_chain_input.roughness_score =
    window ? window->score : std::numeric_limits<double>::quiet_NaN();
  near_chain_input.roughness_score_gate = roughness_score_gate_;
  near_chain_input.minimum_rough_samples = near_chain_min_rough_samples_;
  near_chain_input.recovery_duration_s = near_chain_recovery_duration_s_;
  near_chain_input.loop_closing_suppression_enabled =
    disable_loop_closing_during_near_chain_gate_;
  near_chain_input.loop_closing_cooldown_s =
    near_chain_loop_closing_cooldown_s_;
  near_chain_input.baseline_link_match_threshold =
    mapper->getParamLinkMatchMinimumResponseFine();
  near_chain_input.gated_link_match_threshold =
    near_chain_gated_min_response_;

  RoughnessNearChainGateDecision near_chain_decision;
  RoughnessSpatialRevisitDecision spatial_revisit_decision;
  RoughnessSpatialPassageDecision spatial_passage_decision;
  std::size_t spatial_zone_count = 0;
  bool remember_pose_graph_anchor_after_process = false;
  const char * spatial_pose_source = "odom";
  {
    std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
    near_chain_input.last_scan_stamp_ns = near_chain_last_scan_stamp_ns_;
    near_chain_input.recovery_until_stamp_ns =
      near_chain_recovery_until_stamp_ns_;
    near_chain_input.loop_closing_suppression_until_stamp_ns =
      near_chain_loop_closing_suppression_until_stamp_ns_;
    karto::Pose2 current_spatial_pose = range_scan->GetOdometricPose();
    bool spatial_query_pose_valid =
      std::isfinite(current_spatial_pose.GetX()) &&
      std::isfinite(current_spatial_pose.GetY()) &&
      std::isfinite(current_spatial_pose.GetHeading());
    const bool spatial_pose_epoch_valid =
      processor_type_ == toolbox_types::PROCESS;
    if (!spatial_pose_epoch_valid) {
      roughness_spatial_zones_.clear();
      roughness_pose_graph_anchors_.clear();
      near_chain_recovery_until_stamp_ns_ = 0;
      near_chain_loop_closing_suppression_until_stamp_ns_ = 0;
      near_chain_input.recovery_until_stamp_ns = 0;
      near_chain_input.loop_closing_suppression_until_stamp_ns = 0;
    }
    std::vector<RoughnessSpatialZone> spatial_zones;
    if (spatial_memory_use_pose_graph_anchors_ && spatial_pose_epoch_valid) {
      spatial_zones = refreshPoseGraphSpatialAnchorsLocked(mapper);
      spatial_pose_source = "pose_graph_empty";
      if (!spatial_zones.empty()) {
        const auto predicted_pose = predictPoseGraphSpatialPoseLocked(
          range_scan.get());
        if (predicted_pose) {
          current_spatial_pose = *predicted_pose;
          spatial_query_pose_valid = true;
          spatial_pose_source = "pose_graph_predicted";
        } else {
          spatial_query_pose_valid = false;
          spatial_pose_source = "pose_graph_unavailable";
          ++spatial_memory_pose_graph_prediction_failure_count_;
        }
      }
    } else {
      spatial_zones = roughness_spatial_zones_;
    }
    const bool timestamp_rollback =
      near_chain_last_scan_stamp_ns_ != 0 &&
      stamp_ns < near_chain_last_scan_stamp_ns_;
    if (!timestamp_rollback) {
      spatial_revisit_decision = roughnessSpatialRevisitDecision(
        spatial_pose_epoch_valid &&
        spatial_query_pose_valid &&
        enable_near_chain_gate_ && enable_spatial_memory_gate_,
        current_spatial_pose.GetX(),
        current_spatial_pose.GetY(),
        current_spatial_pose.GetHeading(),
        stamp_ns,
        spatial_zones,
        spatial_memory_radius_m_,
        spatial_memory_min_revisit_separation_s_,
        spatial_memory_min_heading_delta_rad_);
      spatial_passage_decision = roughnessSpatialPassageDecision(
        spatial_pose_epoch_valid &&
        spatial_query_pose_valid &&
        enable_near_chain_gate_ &&
        enable_spatial_memory_gate_ &&
        enable_spatial_passage_hysteresis_,
        current_spatial_pose.GetX(),
        current_spatial_pose.GetY(),
        stamp_ns,
        spatial_zones,
        spatial_passage_radius_m_,
        spatial_passage_max_age_s_);
    }
    near_chain_input.spatial_revisit = spatial_revisit_decision.matched;
    near_chain_input.spatial_passage = spatial_passage_decision.matched;
    near_chain_input.extend_recovery_on_spatial_trigger =
      enable_spatial_passage_hysteresis_;
    near_chain_decision = roughnessNearChainGateDecision(near_chain_input);
    if (
      near_chain_decision.reason ==
      RoughnessNearChainGateReason::kTimestampRollback)
    {
      roughness_spatial_zones_.clear();
      roughness_pose_graph_anchors_.clear();
    } else if (
      spatial_pose_epoch_valid &&
      enable_spatial_memory_gate_ &&
      near_chain_decision.rough_trigger)
    {
      if (spatial_memory_use_pose_graph_anchors_) {
        remember_pose_graph_anchor_after_process = true;
      } else {
        const RoughnessSpatialZone current_zone{
          current_spatial_pose.GetX(),
          current_spatial_pose.GetY(),
          current_spatial_pose.GetHeading(),
          stamp_ns,
          stamp_ns,
        };
        if (
          std::isfinite(current_zone.x_m) &&
          std::isfinite(current_zone.y_m) &&
          std::isfinite(current_zone.heading_rad) &&
          current_zone.rough_stamp_ns > 0 &&
          current_zone.last_rough_stamp_ns >= current_zone.rough_stamp_ns)
        {
          rememberRoughZoneLocked(current_zone);
        } else {
          ++spatial_memory_invalid_pose_count_;
        }
      }
    }
    near_chain_last_scan_stamp_ns_ =
      near_chain_decision.next_last_scan_stamp_ns;
    near_chain_recovery_until_stamp_ns_ =
      near_chain_decision.next_recovery_until_stamp_ns;
    near_chain_loop_closing_suppression_until_stamp_ns_ =
      near_chain_decision.next_loop_closing_suppression_until_stamp_ns;
    spatial_zone_count = spatial_memory_use_pose_graph_anchors_ ?
      roughness_pose_graph_anchors_.size() : roughness_spatial_zones_.size();
  }

  if (near_chain_input.enabled) {
    ++near_chain_diag_count_;
    switch (near_chain_decision.reason) {
      case RoughnessNearChainGateReason::kRough:
        ++near_chain_rough_gate_count_;
        break;
      case RoughnessNearChainGateReason::kSpatialRevisit:
        ++near_chain_spatial_revisit_gate_count_;
        break;
      case RoughnessNearChainGateReason::kSpatialPassage:
        ++near_chain_spatial_passage_gate_count_;
        break;
      case RoughnessNearChainGateReason::kRecovery:
        ++near_chain_recovery_gate_count_;
        break;
      case RoughnessNearChainGateReason::kIncomplete:
      case RoughnessNearChainGateReason::kInvalidInput:
        ++near_chain_incomplete_fail_open_count_;
        break;
      case RoughnessNearChainGateReason::kTimestampRollback:
        ++near_chain_timestamp_reset_count_;
        break;
      case RoughnessNearChainGateReason::kDisabled:
      case RoughnessNearChainGateReason::kNormal:
        break;
    }
    if (near_chain_decision.loop_closing_cooldown) {
      ++near_chain_loop_closing_cooldown_count_;
    }
  }

  AdaptiveRoughnessMemoryState adaptive_memory_state;
  if (roughness_memory_observer_requested) {
    const karto::Pose2 memory_pose = range_scan->GetOdometricPose();
    AdaptiveRoughnessMemoryObservation memory_observation;
    memory_observation.stamp_ns = stamp_ns;
    memory_observation.pose_available =
      std::isfinite(memory_pose.GetX()) &&
      std::isfinite(memory_pose.GetY());
    memory_observation.x_m = memory_pose.GetX();
    memory_observation.y_m = memory_pose.GetY();
    memory_observation.roughness_available =
      window && window->complete && std::isfinite(window->score);
    memory_observation.roughness_score = window ?
      window->score : std::numeric_limits<double>::quiet_NaN();
    {
      std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
      adaptive_memory_state = adaptive_roughness_memory_.update(
        memory_observation);
      if (adaptive_memory_state.timestamp_reset) {
        near_chain_observer_parity_candidate_streak_ = 0;
        near_chain_observer_parity_trusted_ = false;
      }
    }
  }

  ScanMatchInnovation sequential_preview;
  if (
    (near_chain_decision.active || near_chain_decision.loop_closing_cooldown ||
    exact_candidate_observer_requested) &&
    processor_type_ == toolbox_types::PROCESS &&
    mapper->getParamUseScanMatching())
  {
    sequential_preview = previewLocalizedScanMatchInnovationLocked(
      range_scan.get(), mapper, nullptr, nullptr);
  }

  KartoNearChainObservationBatch exact_candidate_batch;
  std::vector<NearChainQuality> exact_candidate_qualities;
  std::vector<AdaptiveConstraintDecision> exact_candidate_decisions;
  NearChainRotationBatchDecision rotation_consistency_decision;
  std::vector<RoughnessConstraintAction> exact_candidate_effective_actions;
  DynamicResponseThresholdDecision dynamic_threshold_decision;
  double requested_link_match_threshold = legacy_fixed_memory_mode ?
    near_chain_decision.requested_threshold :
    mapper->getParamLinkMatchMinimumResponseFine();
  bool adaptive_threshold_intervention_allowed = false;
  bool adaptive_parity_fail_open = false;
  if (
    exact_candidate_observer_requested &&
    processor_type_ == toolbox_types::PROCESS &&
    mapper->getParamUseScanMatching())
  {
    const std::string sensor_name = range_scan->GetSensorName().ToString();
    karto::LocalizedRangeScan * observer_previous_scan = nullptr;
    karto::LocalizedRangeScanVector observer_running_scans;
    const auto previous_scan = preview_last_scans_.find(sensor_name);
    if (previous_scan != preview_last_scans_.end()) {
      observer_previous_scan = previous_scan->second;
    }
    const auto running_scans = preview_running_scans_.find(sensor_name);
    if (running_scans != preview_running_scans_.end()) {
      observer_running_scans = running_scans->second;
    }
    if (observer_previous_scan && !sequential_preview.available) {
      exact_candidate_batch.failure_reason =
        "sequential_prediction_unavailable";
    } else {
      try {
        exact_candidate_batch = observeKartoNearChainCandidates(
          mapper,
          range_scan.get(),
          observer_previous_scan,
          observer_running_scans);
      } catch (const karto::Exception & ex) {
        exact_candidate_batch.failure_reason =
          "karto_exception:" + ex.GetErrorMessage();
      } catch (const std::exception & ex) {
        exact_candidate_batch.failure_reason =
          "exception:" + std::string(ex.what());
      } catch (...) {
        exact_candidate_batch.failure_reason = "unknown_exception";
      }
    }

    if (exact_candidate_batch.available) {
      if (rotation_consistency_config_.enabled) {
        NearChainRotationBatchObservation rotation_observation;
        rotation_observation.stamp_ns = stamp_ns;
        rotation_observation.target_state_id =
          exact_candidate_batch.predicted_target_state_id;
        rotation_observation.memory_available = adaptive_memory_state.available;
        rotation_observation.memory_strength = adaptive_memory_state.strength;
        rotation_observation.candidates.reserve(
          exact_candidate_batch.candidates.size());
        for (const auto & candidate : exact_candidate_batch.candidates) {
          NearChainRotationCandidateObservation observation;
          observation.eligible = candidate.chain_size_accepted;
          observation.normalized_response = candidate.normalized_response;
          observation.translation_disagreement_m =
            candidate.translation_disagreement_m;
          observation.yaw_disagreement_rad = candidate.yaw_disagreement_rad;
          observation.covariance_weighted_disagreement_sq =
            candidate.covariance_weighted_disagreement_sq;
          observation.chain_state_ids = candidate.chain_state_ids;
          rotation_observation.candidates.push_back(std::move(observation));
        }
        const bool rotation_intervention_enabled =
          roughness_constraint_mode_ ==
          RoughnessConstraintMode::kAdaptiveQualityAware;
        {
          std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
          rotation_consistency_decision =
            near_chain_rotation_consistency_.evaluate(
            rotation_observation, rotation_intervention_enabled);
        }
        ++near_chain_rotation_batch_count_;
        if (rotation_consistency_decision.timestamp_reset) {
          ++near_chain_rotation_timestamp_reset_count_;
        }
        if (rotation_consistency_decision.fail_open) {
          ++near_chain_rotation_fail_open_count_;
        }
        for (const auto & decision : rotation_consistency_decision.candidates) {
          if (decision.candidate_risky) {
            ++near_chain_rotation_risky_candidate_count_;
          }
          if (decision.proposed_reject) {
            ++near_chain_rotation_proposed_reject_count_;
          }
        }
      }
      exact_candidate_qualities.reserve(
        exact_candidate_batch.candidates.size());
      exact_candidate_decisions.reserve(
        exact_candidate_batch.candidates.size());
      exact_candidate_effective_actions.reserve(
        exact_candidate_batch.candidates.size());
      std::vector<DynamicResponseThresholdCandidate> threshold_candidates;
      threshold_candidates.reserve(exact_candidate_batch.candidates.size());
      for (std::size_t index = 0;
        index < exact_candidate_batch.candidates.size(); ++index)
      {
        const auto & candidate = exact_candidate_batch.candidates[index];
        const NearChainQuality quality = evaluateNearChainQuality(
          candidate.qualityObservation(), near_chain_quality_config_);
        const RoughnessConstraintMode decision_mode =
          roughness_constraint_mode_ ==
          RoughnessConstraintMode::kAdaptiveMemoryObserve ?
          RoughnessConstraintMode::kAdaptiveQualityAware :
          roughness_constraint_mode_;
        const AdaptiveConstraintDecision decision = adaptiveConstraintDecision(
          decision_mode,
          adaptive_memory_state,
          quality,
          adaptive_constraint_policy_config_);
        RoughnessConstraintAction effective_action = decision.action;
        const bool use_rotation_shadow =
          roughness_constraint_mode_ ==
          RoughnessConstraintMode::kInterventionOff ||
          roughness_constraint_mode_ ==
          RoughnessConstraintMode::kAdaptiveMemoryObserve ||
          roughness_constraint_mode_ ==
          RoughnessConstraintMode::kAdaptiveQualityAware;
        if (
          use_rotation_shadow &&
          index < rotation_consistency_decision.candidates.size() &&
          rotation_consistency_decision.candidates[index].proposed_reject &&
          !rotation_consistency_decision.candidates[index].fail_open)
        {
          effective_action = RoughnessConstraintAction::kRejected;
        }
        exact_candidate_qualities.push_back(quality);
        exact_candidate_decisions.push_back(decision);
        exact_candidate_effective_actions.push_back(effective_action);
        threshold_candidates.push_back(
          DynamicResponseThresholdCandidate{
            candidate.chain_size_accepted,
            candidate.normalized_response,
            effective_action});
      }
      dynamic_threshold_decision = deriveDynamicResponseThreshold(
        exact_candidate_batch.baseline_response_threshold,
        adaptive_dynamic_threshold_max_,
        kParameterComparisonTolerance + 1.0e-6,
        threshold_candidates);

      const bool intervention_mode =
        roughness_constraint_mode_ ==
        RoughnessConstraintMode::kAdaptiveMemoryOnly ||
        roughness_constraint_mode_ ==
        RoughnessConstraintMode::kAdaptiveQualityAware;
      if (
        intervention_mode &&
        dynamic_threshold_decision.intervention_required &&
        !exact_candidate_batch.candidate_endpoints_unique)
      {
        dynamic_threshold_decision.fail_open = true;
        ++near_chain_adaptive_fail_open_count_;
      }
      if (
        intervention_mode &&
        dynamic_threshold_decision.intervention_required &&
        !dynamic_threshold_decision.fail_open)
      {
        std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
        adaptive_threshold_intervention_allowed =
          near_chain_observer_parity_trusted_;
        if (adaptive_threshold_intervention_allowed) {
          requested_link_match_threshold =
            dynamic_threshold_decision.requested_threshold;
        } else {
          adaptive_parity_fail_open = true;
          ++near_chain_adaptive_fail_open_count_;
        }
      }
    } else {
      ++near_chain_exact_observer_unavailable_count_;
      std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
      near_chain_observer_parity_candidate_streak_ = 0;
      near_chain_observer_parity_trusted_ = false;
      near_chain_rotation_consistency_.reset();
    }
  }

  std::size_t edge_count_before = 0;
  if (mapper->GetGraph()) {
    edge_count_before = mapper->GetGraph()->GetEdges().size();
  }
  KartoLoopClosureObserver loop_closure_observer;
  if (
    enable_loop_closure_candidate_observer_ &&
    processor_type_ == toolbox_types::PROCESS &&
    mapper->getParamUseScanMatching())
  {
    (void)loop_closure_observer.attach(mapper, range_scan.get(), stamp_ns);
  }

  const bool apply_threshold_override =
    (near_chain_decision.active ||
    adaptive_threshold_intervention_allowed) &&
    processor_type_ == toolbox_types::PROCESS &&
    mapper->getParamUseScanMatching() &&
    std::isfinite(mapper->getParamLinkMatchMinimumResponseFine()) &&
    std::isfinite(requested_link_match_threshold) &&
    requested_link_match_threshold >
    mapper->getParamLinkMatchMinimumResponseFine() +
    kParameterComparisonTolerance;
  ScopedLinkMatchMinimumResponse threshold_override(
    mapper,
    apply_threshold_override,
    requested_link_match_threshold);
  if (threshold_override.overridden()) {
    ++near_chain_threshold_override_count_;
  }
  ScopedLoopClosingSuppression loop_closing_suppression(
    mapper,
    near_chain_decision.suppress_loop_closing &&
    processor_type_ == toolbox_types::PROCESS &&
    disable_loop_closing_during_near_chain_gate_);
  if (loop_closing_suppression.suppressed()) {
    ++near_chain_loop_closing_suppression_count_;
  }

  bool processed = false;
  bool update_reprocessing_transform = false;
  karto::Matrix3 covariance;
  covariance.SetToIdentity();

  try {
    if (processor_type_ == toolbox_types::PROCESS) {
      processed = mapper->Process(range_scan.get(), &covariance);
    } else if (processor_type_ == toolbox_types::PROCESS_FIRST_NODE) {
      processed = mapper->ProcessAtDock(range_scan.get(), &covariance);
      processor_type_ = toolbox_types::PROCESS;
      update_reprocessing_transform = true;
    } else if (processor_type_ == toolbox_types::PROCESS_NEAR_REGION) {
      boost::mutex::scoped_lock pose_lock(pose_mutex_);
      if (!process_near_pose_) {
        RCLCPP_ERROR(
          get_logger(),
          "Process near region called without a valid region request. Ignoring scan.");
        const bool restored = threshold_override.restore();
        if (!restored) {
          ++near_chain_threshold_restore_failure_count_;
        }
        if (!loop_closing_suppression.restore()) {
          ++near_chain_loop_closing_restore_failure_count_;
        }
        return nullptr;
      }
      range_scan->SetOdometricPose(*process_near_pose_);
      range_scan->SetCorrectedPose(range_scan->GetOdometricPose());
      process_near_pose_.reset(nullptr);
      processed = mapper->ProcessAgainstNodesNearBy(
        range_scan.get(), false, &covariance);
      update_reprocessing_transform = true;
      processor_type_ = toolbox_types::PROCESS;
    } else {
      RCLCPP_FATAL(
        get_logger(),
        "RoughnessAwareSlamToolbox: no valid processor type set.");
      std::exit(-1);
    }
  } catch (...) {
    if (!threshold_override.restore()) {
      ++near_chain_threshold_restore_failure_count_;
    }
    if (!loop_closing_suppression.restore()) {
      ++near_chain_loop_closing_restore_failure_count_;
    }
    throw;
  }

  KartoLoopClosureObservationBatch loop_closure_batch;
  if (enable_loop_closure_candidate_observer_) {
    loop_closure_batch = loop_closure_observer.finish();
  }

  const bool threshold_restore_succeeded = threshold_override.restore();
  if (!threshold_restore_succeeded) {
    ++near_chain_threshold_restore_failure_count_;
    RCLCPP_ERROR(
      get_logger(),
      "Failed to restore Karto LinkMatchMinimumResponseFine after scan %lld.",
      static_cast<long long>(stamp_ns));
  }
  const bool loop_closing_restore_succeeded =
    loop_closing_suppression.restore();
  if (!loop_closing_restore_succeeded) {
    ++near_chain_loop_closing_restore_failure_count_;
    RCLCPP_ERROR(
      get_logger(),
      "Failed to restore Karto DoLoopClosing after scan %lld.",
      static_cast<long long>(stamp_ns));
  }

  if (processed && roughness_memory_observer_requested) {
    RoughnessScanMemorySample sample;
    sample.mapper_generation = mapper_generation_.load();
    sample.state_id = range_scan->GetStateId();
    sample.unique_id = range_scan->GetUniqueId();
    sample.stamp_ns = stamp_ns;
    sample.available = adaptive_memory_state.available;
    sample.rough_trigger = adaptive_memory_state.rough_trigger;
    sample.roughness_score_available =
      window && window->complete && std::isfinite(window->score);
    sample.roughness_score = window ?
      window->score : std::numeric_limits<double>::quiet_NaN();
    sample.memory_strength = adaptive_memory_state.strength;
    sample.effective_duration_s = adaptive_memory_state.effective_duration_s;
    sample.effective_distance_m = adaptive_memory_state.effective_distance_m;
    std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
    if (adaptive_memory_state.timestamp_reset) {
      roughness_loop_memory_.reset(sample.mapper_generation);
    }
    (void)roughness_loop_memory_.record(sample);
  }

  std::size_t edge_count_after = edge_count_before;
  std::size_t new_edge_count = 0;
  std::size_t new_nonsequential_edge_count = 0;
  if (mapper->GetGraph()) {
    const auto & edges = mapper->GetGraph()->GetEdges();
    edge_count_after = edges.size();
    if (edge_count_after >= edge_count_before) {
      new_edge_count = edge_count_after - edge_count_before;
      for (std::size_t index = edge_count_before; index < edge_count_after; ++index) {
        const auto * edge = edges[index];
        if (
          !edge || !edge->GetSource() || !edge->GetTarget() ||
          !edge->GetSource()->GetObject() || !edge->GetTarget()->GetObject())
        {
          continue;
        }
        const std::int32_t source_id =
          edge->GetSource()->GetObject()->GetStateId();
        const std::int32_t target_id =
          edge->GetTarget()->GetObject()->GetStateId();
        if (std::abs(source_id - target_id) > 1) {
          ++new_nonsequential_edge_count;
        }
      }
    }
  }

  std::vector<std::int32_t> loop_closure_source_unique_ids;
  std::vector<RoughnessLoopMemoryPrior> loop_closure_memory_priors;
  if (enable_loop_closure_candidate_observer_ && processed) {
    updateKartoLoopClosureOutcomes(
      loop_closure_batch,
      mapper,
      range_scan.get());
    loop_closure_candidate_count_.fetch_add(loop_closure_batch.attempts.size());
    loop_closure_accepted_count_.fetch_add(
      loop_closure_batch.accepted_callback_count);
    loop_closure_actual_edge_count_.fetch_add(
      loop_closure_batch.actual_loop_edge_count);
    loop_closure_accepted_without_edge_count_.fetch_add(
      loop_closure_batch.accepted_without_edge_count);
    if (loop_closure_batch.failure_reason == "graph_not_initialized") {
      ++loop_closure_graph_warmup_count_;
    } else if (!loop_closure_batch.edge_pairing_complete) {
      ++loop_closure_pairing_failure_count_;
    }
    loop_closure_memory_priors.reserve(loop_closure_batch.attempts.size());
    {
      std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
      for (const auto & attempt : loop_closure_batch.attempts) {
        loop_closure_memory_priors.push_back(
          roughness_loop_memory_.evaluate(
            mapper_generation_.load(),
            attempt.chain_unique_ids,
            attempt.source_unique_id,
            loop_closure_batch.target_unique_id));
      }
    }
    for (const auto & attempt : loop_closure_batch.attempts) {
      if (attempt.actual_edge_present && attempt.source_unique_id >= 0) {
        loop_closure_source_unique_ids.push_back(attempt.source_unique_id);
      }
    }
  }

  if (exact_candidate_observer_requested && exact_candidate_batch.available) {
    if (processed) {
      updateKartoNearChainCandidateOutcomes(
        exact_candidate_batch,
        mapper,
        range_scan.get(),
        threshold_override.applied(),
        edge_count_before,
        loop_closure_source_unique_ids);
      near_chain_exact_candidate_count_.fetch_add(
        exact_candidate_batch.candidates.size());
      for (std::size_t index = 0;
        index < exact_candidate_batch.candidates.size(); ++index)
      {
        if (
          index < exact_candidate_qualities.size() &&
          exact_candidate_qualities[index].available)
        {
          ++near_chain_quality_available_count_;
        }
        const AdaptiveConstraintDecision decision =
          index < exact_candidate_decisions.size() ?
          exact_candidate_decisions[index] : AdaptiveConstraintDecision{};
        const NearChainRotationCandidateDecision rotation_decision =
          index < rotation_consistency_decision.candidates.size() ?
          rotation_consistency_decision.candidates[index] :
          NearChainRotationCandidateDecision{};
        const bool candidate_fail_open =
          decision.fail_open || rotation_decision.fail_open ||
          (dynamic_threshold_decision.intervention_required &&
          dynamic_threshold_decision.fail_open) ||
          adaptive_parity_fail_open;
        if (candidate_fail_open) {
          ++near_chain_adaptive_fail_open_count_;
        }
        const RoughnessConstraintAction effective_action =
          index < exact_candidate_effective_actions.size() ?
          exact_candidate_effective_actions[index] : decision.action;
        switch (effective_action) {
          case RoughnessConstraintAction::kAccepted:
            ++near_chain_adaptive_accept_count_;
            break;
          case RoughnessConstraintAction::kDownweighted:
            ++near_chain_adaptive_downweight_shadow_count_;
            break;
          case RoughnessConstraintAction::kRejected:
            ++near_chain_adaptive_reject_count_;
            break;
        }
        if (
          rotation_decision.intervention_reject &&
          adaptive_threshold_intervention_allowed &&
          !exact_candidate_batch.candidates[index].applied_response_accepted)
        {
          ++near_chain_rotation_applied_reject_count_;
        }
      }
    }
    std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
    if (
      processed && exact_candidate_batch.outcome_parity &&
      exact_candidate_batch.eligible_chain_count > 0U)
    {
      near_chain_observer_parity_candidate_streak_ +=
        exact_candidate_batch.eligible_chain_count;
      near_chain_observer_parity_trusted_ =
        near_chain_observer_parity_candidate_streak_ >=
        near_chain_observer_min_parity_candidates_;
    } else if (
      processed &&
      exact_candidate_batch.eligible_chain_count > 0U &&
      !exact_candidate_batch.outcome_parity)
    {
      ++near_chain_exact_parity_failure_count_;
      near_chain_observer_parity_candidate_streak_ = 0;
      near_chain_observer_parity_trusted_ = false;
      near_chain_rotation_consistency_.reset();
    } else if (
      !processed && exact_candidate_batch.eligible_chain_count > 0U)
    {
      near_chain_observer_parity_candidate_streak_ = 0;
      near_chain_observer_parity_trusted_ = false;
      near_chain_rotation_consistency_.reset();
    }
  }

  double process_delta_x_m = std::numeric_limits<double>::quiet_NaN();
  double process_delta_y_m = std::numeric_limits<double>::quiet_NaN();
  double process_delta_translation_m =
    std::numeric_limits<double>::quiet_NaN();
  double process_delta_yaw_rad = std::numeric_limits<double>::quiet_NaN();
  if (processed && sequential_preview.available) {
    const karto::Pose2 final_pose = range_scan->GetCorrectedPose();
    process_delta_x_m = final_pose.GetX() - sequential_preview.matched_x_m;
    process_delta_y_m = final_pose.GetY() - sequential_preview.matched_y_m;
    process_delta_translation_m =
      std::hypot(process_delta_x_m, process_delta_y_m);
    process_delta_yaw_rad = normalizedAngle(
      final_pose.GetHeading() - sequential_preview.matched_yaw_rad);
  }

  karto::LocalizedRangeScan * processed_scan = nullptr;
  if (processed) {
    processed_scan = range_scan.get();
    if (enable_interactive_mode_) {
      scan_holder_->addScan(*scan);
    }
    setTransformFromPoses(
      processed_scan->GetCorrectedPose(),
      odom_pose,
      scan->header.stamp,
      update_reprocessing_transform);
    dataset_->Add(processed_scan);
    publishPose(
      processed_scan->GetCorrectedPose(),
      covariance,
      scan->header.stamp);
    trackProcessedScanLocked(processed_scan, mapper);
    updateRoughnessPlaceRecognitionLocked(
      processed_scan, stamp_ns, window);
    if (spatial_memory_use_pose_graph_anchors_) {
      std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
      if (remember_pose_graph_anchor_after_process) {
        rememberPoseGraphSpatialAnchorLocked(processed_scan, stamp_ns);
      }
      (void)refreshPoseGraphSpatialAnchorsLocked(mapper);
      spatial_zone_count = roughness_pose_graph_anchors_.size();
    }
    range_scan.release();
  }

  if (near_chain_decision.active) {
    near_chain_new_edge_count_.fetch_add(new_edge_count);
    near_chain_new_nonsequential_edge_count_.fetch_add(
      new_nonsequential_edge_count);
  }

  if (exact_candidate_observer_requested) {
    std::size_t observer_parity_streak = 0;
    bool observer_parity_trusted = false;
    {
      std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
      observer_parity_streak = near_chain_observer_parity_candidate_streak_;
      observer_parity_trusted = near_chain_observer_parity_trusted_;
    }
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_ADAPTIVE_MEMORY stamp_ns=%lld mapper_generation=%llu "
      "mode=%s scan_processed=%s available=%s active=%s rough_trigger=%s "
      "timestamp_reset=%s "
      "strength=%.9g event_intensity=%.9g elapsed_s=%.9g "
      "distance_m=%.9g stable_recovery_s=%.9g "
      "effective_duration_s=%.9g effective_distance_m=%.9g "
      "observer_available=%s observer_failure=%s preview_candidates=%zu "
      "exact_candidates=%zu "
      "eligible_candidates=%zu baseline_pose_fusion=%zu "
      "applied_pose_fusion=%zu actual_candidate_edges=%zu "
      "candidate_endpoints_unique=%s new_edge_accounting_complete=%s "
      "new_edges=%zu classified_new_edges=%zu unexpected_new_edges=%zu "
      "outcome_parity=%s parity_streak=%zu parity_trusted=%s "
      "dynamic_available=%s dynamic_fail_open=%s "
      "dynamic_intervention_required=%s intervention_allowed=%s "
      "parity_fail_open=%s "
      "dynamic_rejected_candidates=%zu dynamic_preserved_candidates=%zu "
      "highest_rejected_response=%.9g lowest_preserved_response=%.9g "
      "baseline_threshold=%.9g requested_threshold=%.9g "
      "applied_threshold=%.9g soft_weighting_available=false "
      "rotation_enabled=%s rotation_available=%s "
      "rotation_timestamp_reset=%s rotation_model_available=%s "
      "rotation_memory_trigger=%s rotation_repeated_chain_trigger=%s "
      "rotation_latch_before=%s rotation_latch_after=%s "
      "rotation_fail_open=%s rotation_history_count=%zu "
      "rotation_yaw_median_rad=%.9g rotation_yaw_scale_rad=%.9g "
      "rotation_yaw_limit_rad=%.9g rotation_yaw_extreme_limit_rad=%.9g "
      "rotation_translation_median_m=%.9g "
      "rotation_translation_scale_m=%.9g "
      "rotation_translation_limit_m=%.9g "
      "rotation_weighted_median=%.9g rotation_weighted_scale=%.9g "
      "rotation_weighted_limit=%.9g",
      static_cast<long long>(stamp_ns),
      static_cast<unsigned long long>(mapper_generation_.load()),
      roughnessConstraintModeName(roughness_constraint_mode_),
      processed ? "true" : "false",
      adaptive_memory_state.available ? "true" : "false",
      adaptive_memory_state.active ? "true" : "false",
      adaptive_memory_state.rough_trigger ? "true" : "false",
      adaptive_memory_state.timestamp_reset ? "true" : "false",
      adaptive_memory_state.strength,
      adaptive_memory_state.event_intensity,
      adaptive_memory_state.elapsed_s,
      adaptive_memory_state.distance_m,
      adaptive_memory_state.stable_recovery_s,
      adaptive_memory_state.effective_duration_s,
      adaptive_memory_state.effective_distance_m,
      exact_candidate_batch.available ? "true" : "false",
      exact_candidate_batch.failure_reason.empty() ? "none" :
      exact_candidate_batch.failure_reason.c_str(),
      exact_candidate_batch.candidates.size(),
      processed ? exact_candidate_batch.candidates.size() : 0U,
      exact_candidate_batch.eligible_chain_count,
      exact_candidate_batch.baseline_pose_fusion_count,
      exact_candidate_batch.applied_pose_fusion_count,
      exact_candidate_batch.actual_candidate_edge_count,
      exact_candidate_batch.candidate_endpoints_unique ? "true" : "false",
      exact_candidate_batch.new_edge_accounting_complete ? "true" : "false",
      exact_candidate_batch.new_edge_count,
      exact_candidate_batch.classified_new_edge_count,
      exact_candidate_batch.unexpected_new_edge_count,
      exact_candidate_batch.outcome_parity ? "true" : "false",
      observer_parity_streak,
      observer_parity_trusted ? "true" : "false",
      dynamic_threshold_decision.available ? "true" : "false",
      dynamic_threshold_decision.fail_open ? "true" : "false",
      dynamic_threshold_decision.intervention_required ? "true" : "false",
      adaptive_threshold_intervention_allowed ? "true" : "false",
      adaptive_parity_fail_open ? "true" : "false",
      dynamic_threshold_decision.rejected_candidate_count,
      dynamic_threshold_decision.preserved_candidate_count,
      dynamic_threshold_decision.highest_rejected_response,
      dynamic_threshold_decision.lowest_preserved_response,
      exact_candidate_batch.baseline_response_threshold,
      requested_link_match_threshold,
      threshold_override.applied(),
      rotation_consistency_config_.enabled ? "true" : "false",
      rotation_consistency_decision.available ? "true" : "false",
      rotation_consistency_decision.timestamp_reset ? "true" : "false",
      rotation_consistency_decision.model_available ? "true" : "false",
      rotation_consistency_decision.memory_trigger ? "true" : "false",
      rotation_consistency_decision.repeated_chain_trigger ? "true" : "false",
      rotation_consistency_decision.latch_active_before ? "true" : "false",
      rotation_consistency_decision.latch_active_after ? "true" : "false",
      rotation_consistency_decision.fail_open ? "true" : "false",
      rotation_consistency_decision.history_count,
      rotation_consistency_decision.yaw_limit.median,
      rotation_consistency_decision.yaw_limit.scale,
      rotation_consistency_decision.yaw_limit.limit,
      rotation_consistency_decision.yaw_limit.extreme_limit,
      rotation_consistency_decision.translation_limit.median,
      rotation_consistency_decision.translation_limit.scale,
      rotation_consistency_decision.translation_limit.limit,
      rotation_consistency_decision.weighted_disagreement_limit.median,
      rotation_consistency_decision.weighted_disagreement_limit.scale,
      rotation_consistency_decision.weighted_disagreement_limit.limit);

    for (std::size_t index = 0;
      processed && index < exact_candidate_batch.candidates.size(); ++index)
    {
      const auto & candidate = exact_candidate_batch.candidates[index];
      const NearChainQuality quality =
        index < exact_candidate_qualities.size() ?
        exact_candidate_qualities[index] : NearChainQuality{};
      const AdaptiveConstraintDecision decision =
        index < exact_candidate_decisions.size() ?
        exact_candidate_decisions[index] : AdaptiveConstraintDecision{};
      const NearChainRotationCandidateDecision rotation_decision =
        index < rotation_consistency_decision.candidates.size() ?
        rotation_consistency_decision.candidates[index] :
        NearChainRotationCandidateDecision{};
      const RoughnessConstraintAction effective_action =
        index < exact_candidate_effective_actions.size() ?
        exact_candidate_effective_actions[index] : decision.action;
      std::ostringstream chain_state_ids;
      std::ostringstream chain_unique_ids;
      for (std::size_t chain_index = 0;
        chain_index < candidate.chain_state_ids.size(); ++chain_index)
      {
        if (chain_index != 0U) {
          chain_state_ids << ',';
          chain_unique_ids << ',';
        }
        chain_state_ids << candidate.chain_state_ids[chain_index];
        chain_unique_ids << candidate.chain_unique_ids[chain_index];
      }
      const std::string chain_state_ids_text = chain_state_ids.str();
      const std::string chain_unique_ids_text = chain_unique_ids.str();
      std::string sensor_name_token = candidate.sensor_name;
      std::replace_if(
        sensor_name_token.begin(), sensor_name_token.end(),
        [](const unsigned char value) {return std::isspace(value) != 0;},
        '_');
      const bool inserted_candidate_edge =
        candidate.actual_edge_present &&
        !candidate.prelinked_by_sequential_or_running;
      const bool response_rejected =
        candidate.chain_size_accepted &&
        !candidate.applied_response_accepted;
      const char * backend_action = "accepted";
      if (!candidate.chain_size_accepted) {
        backend_action = "ineligible_chain";
      } else if (response_rejected) {
        backend_action = "rejected_response";
      } else if (!candidate.distance_accepted) {
        backend_action = "pose_fused_distance_rejected";
      } else if (candidate.prelinked_by_sequential_or_running) {
        backend_action = "pose_fused_duplicate_edge";
      } else if (inserted_candidate_edge) {
        backend_action = "pose_fused_edge_inserted";
      } else {
        backend_action = "pose_fused_edge_missing";
      }
      const bool threshold_fail_open =
        dynamic_threshold_decision.intervention_required &&
        dynamic_threshold_decision.fail_open;
      const bool candidate_fail_open =
        decision.fail_open || rotation_decision.fail_open ||
        threshold_fail_open || adaptive_parity_fail_open;
      double candidate_vs_odom_prediction_translation_m =
        std::numeric_limits<double>::quiet_NaN();
      double candidate_vs_odom_prediction_yaw_rad =
        std::numeric_limits<double>::quiet_NaN();
      if (sequential_preview.available && candidate.chain_size_accepted) {
        candidate_vs_odom_prediction_translation_m = std::hypot(
          candidate.matched_target_pose.GetX() -
          sequential_preview.predicted_x_m,
          candidate.matched_target_pose.GetY() -
          sequential_preview.predicted_y_m);
        candidate_vs_odom_prediction_yaw_rad = std::fabs(
          normalizedAngle(
            candidate.matched_target_pose.GetHeading() -
            sequential_preview.predicted_yaw_rad));
      }
      RCLCPP_INFO(
        get_logger(),
        "ROUGHNESS_NEAR_CHAIN_CANDIDATE stamp_ns=%lld "
        "mapper_generation=%llu candidate_index=%zu edge_type=near_chain "
        "sensor=%s source_state_id=%d source_unique_id=%d "
        "target_state_id=%d target_unique_id=%d chain_size=%zu "
        "chain_state_ids=%s chain_unique_ids=%s "
        "normalized_response=%.9g matched_x_m=%.9g matched_y_m=%.9g "
        "matched_yaw_rad=%.9g matched_target_x_m=%.9g "
        "matched_target_y_m=%.9g matched_target_yaw_rad=%.9g "
        "sequential_available=%s sequential_response=%.9g "
        "sequential_predicted_x_m=%.9g sequential_predicted_y_m=%.9g "
        "sequential_predicted_yaw_rad=%.9g sequential_matched_x_m=%.9g "
        "sequential_matched_y_m=%.9g sequential_matched_yaw_rad=%.9g "
        "sequential_translation_m=%.9g sequential_yaw_rad=%.9g "
        "candidate_vs_sequential_translation_m=%.9g "
        "candidate_vs_sequential_yaw_rad=%.9g "
        "candidate_vs_odom_prediction_translation_m=%.9g "
        "candidate_vs_odom_prediction_yaw_rad=%.9g "
        "relative_x_m=%.9g relative_y_m=%.9g "
        "relative_yaw_rad=%.9g covariance_xx=%.9g covariance_xy=%.9g "
        "covariance_xyaw=%.9g covariance_yy=%.9g "
        "covariance_yyaw=%.9g covariance_yawyaw=%.9g "
        "xy_covariance_max_eigenvalue_m2=%.9g "
        "translation_disagreement_m=%.9g yaw_disagreement_rad=%.9g "
        "scan_match_covariance_weighted_disagreement_sq=%.9g "
        "memory_available=%s memory_strength=%.9g "
        "adaptive_duration_s=%.9g adaptive_distance_m=%.9g "
        "quality_available=%s response_risk=%.9g "
        "disagreement_risk=%.9g covariance_risk=%.9g quality_risk=%.9g "
        "combined_risk=%.9g quality_action=%s effective_action=%s "
        "shadow_action=%s quality_fail_open=%s rotation_fail_open=%s "
        "threshold_fail_open=%s parity_fail_open=%s fail_open=%s "
        "rotation_evidence_available=%s rotation_candidate_risky=%s "
        "rotation_proposed_reject=%s rotation_intervention_reject=%s "
        "rotation_prior_chain_overlap=%.9g "
        "rotation_log1p_weighted_disagreement=%.9g "
        "rotation_model_available=%s rotation_memory_trigger=%s "
        "rotation_repeated_chain_trigger=%s rotation_latch_before=%s "
        "rotation_latch_after=%s rotation_history_count=%zu "
        "rotation_yaw_limit_rad=%.9g rotation_yaw_extreme_limit_rad=%.9g "
        "rotation_translation_limit_m=%.9g "
        "rotation_weighted_disagreement_limit=%.9g "
        "information_scale=%.9g backend_action=%s "
        "chain_size_accepted=%s baseline_response_accepted=%s "
        "applied_response_accepted=%s distance_accepted=%s "
        "prelinked=%s edge_present=%s edge_inserted=%s downweighted=false "
        "baseline_threshold=%.9g applied_threshold=%.9g "
        "pre_residual_x_m=%.9g pre_residual_y_m=%.9g "
        "pre_residual_yaw_rad=%.9g post_residual_x_m=%.9g "
        "post_residual_y_m=%.9g post_residual_yaw_rad=%.9g "
        "post_covariance_weighted_residual_sq=%.9g outcome_parity=%s",
        static_cast<long long>(stamp_ns),
        static_cast<unsigned long long>(mapper_generation_.load()),
        candidate.candidate_index,
        sensor_name_token.c_str(),
        candidate.source_state_id,
        candidate.source_unique_id,
        candidate.target_state_id,
        candidate.target_unique_id,
        candidate.chain_state_ids.size(),
        chain_state_ids_text.empty() ? "none" :
        chain_state_ids_text.c_str(),
        chain_unique_ids_text.empty() ? "none" :
        chain_unique_ids_text.c_str(),
        candidate.normalized_response,
        candidate.matched_sensor_pose.GetX(),
        candidate.matched_sensor_pose.GetY(),
        candidate.matched_sensor_pose.GetHeading(),
        candidate.matched_target_pose.GetX(),
        candidate.matched_target_pose.GetY(),
        candidate.matched_target_pose.GetHeading(),
        sequential_preview.available ? "true" : "false",
        sequential_preview.response,
        sequential_preview.predicted_x_m,
        sequential_preview.predicted_y_m,
        sequential_preview.predicted_yaw_rad,
        sequential_preview.matched_x_m,
        sequential_preview.matched_y_m,
        sequential_preview.matched_yaw_rad,
        sequential_preview.translation_m,
        sequential_preview.yaw_rad,
        candidate.translation_disagreement_m,
        candidate.yaw_disagreement_rad,
        candidate_vs_odom_prediction_translation_m,
        candidate_vs_odom_prediction_yaw_rad,
        candidate.relative_transform.GetX(),
        candidate.relative_transform.GetY(),
        candidate.relative_transform.GetHeading(),
        candidate.factor_covariance(0, 0),
        candidate.factor_covariance(0, 1),
        candidate.factor_covariance(0, 2),
        candidate.factor_covariance(1, 1),
        candidate.factor_covariance(1, 2),
        candidate.factor_covariance(2, 2),
        candidate.xy_covariance_max_eigenvalue_m2,
        candidate.translation_disagreement_m,
        candidate.yaw_disagreement_rad,
        candidate.covariance_weighted_disagreement_sq,
        adaptive_memory_state.available ? "true" : "false",
        adaptive_memory_state.strength,
        adaptive_memory_state.effective_duration_s,
        adaptive_memory_state.effective_distance_m,
        quality.available ? "true" : "false",
        quality.response_risk,
        quality.disagreement_risk,
        quality.covariance_risk,
        quality.risk,
        decision.combined_risk,
        roughnessConstraintActionName(decision.action),
        roughnessConstraintActionName(effective_action),
        roughnessConstraintActionName(effective_action),
        decision.fail_open ? "true" : "false",
        rotation_decision.fail_open ? "true" : "false",
        threshold_fail_open ? "true" : "false",
        adaptive_parity_fail_open ? "true" : "false",
        candidate_fail_open ? "true" : "false",
        rotation_decision.evidence_available ? "true" : "false",
        rotation_decision.candidate_risky ? "true" : "false",
        rotation_decision.proposed_reject ? "true" : "false",
        rotation_decision.intervention_reject ? "true" : "false",
        rotation_decision.prior_chain_overlap,
        rotation_decision.log1p_covariance_weighted_disagreement,
        rotation_consistency_decision.model_available ? "true" : "false",
        rotation_consistency_decision.memory_trigger ? "true" : "false",
        rotation_consistency_decision.repeated_chain_trigger ? "true" : "false",
        rotation_consistency_decision.latch_active_before ? "true" : "false",
        rotation_consistency_decision.latch_active_after ? "true" : "false",
        rotation_consistency_decision.history_count,
        rotation_consistency_decision.yaw_limit.limit,
        rotation_consistency_decision.yaw_limit.extreme_limit,
        rotation_consistency_decision.translation_limit.limit,
        rotation_consistency_decision.weighted_disagreement_limit.limit,
        decision.information_scale,
        backend_action,
        candidate.chain_size_accepted ? "true" : "false",
        candidate.baseline_response_accepted ? "true" : "false",
        candidate.applied_response_accepted ? "true" : "false",
        candidate.distance_accepted ? "true" : "false",
        candidate.prelinked_by_sequential_or_running ? "true" : "false",
        candidate.actual_edge_present ? "true" : "false",
        inserted_candidate_edge ? "true" : "false",
        exact_candidate_batch.baseline_response_threshold,
        exact_candidate_batch.applied_response_threshold,
        candidate.pre_residual_x_m,
        candidate.pre_residual_y_m,
        candidate.pre_residual_yaw_rad,
        candidate.post_residual_x_m,
        candidate.post_residual_y_m,
        candidate.post_residual_yaw_rad,
        candidate.post_covariance_weighted_residual_sq,
        candidate.outcome_parity ? "true" : "false");
    }
  }

  if (enable_loop_closure_candidate_observer_) {
    RCLCPP_INFO(
      get_logger(),
      "KARTO_LOOP_CANDIDATE_BATCH stamp_ns=%lld mapper_generation=%llu "
      "scan_processed=%s observer_available=%s listener_attached=%s "
      "listener_detached=%s single_sensor=%s candidate_identity_complete=%s "
      "listener_sequence_complete=%s edge_pairing_complete=%s attempts=%zu "
      "accepted_callbacks=%zu actual_loop_edges=%zu accepted_without_edges=%zu "
      "ambiguous_loop_edges=%zu "
      "unpaired_loop_edges=%zu failure=%s",
      static_cast<long long>(stamp_ns),
      static_cast<unsigned long long>(mapper_generation_.load()),
      processed ? "true" : "false",
      loop_closure_batch.available ? "true" : "false",
      loop_closure_batch.listener_attached ? "true" : "false",
      loop_closure_batch.listener_detached ? "true" : "false",
      loop_closure_batch.single_sensor_context ? "true" : "false",
      loop_closure_batch.candidate_identity_complete ? "true" : "false",
      loop_closure_batch.listener_sequence_complete ? "true" : "false",
      loop_closure_batch.edge_pairing_complete ? "true" : "false",
      loop_closure_batch.attempts.size(),
      loop_closure_batch.accepted_callback_count,
      loop_closure_batch.actual_loop_edge_count,
      loop_closure_batch.accepted_without_edge_count,
      loop_closure_batch.ambiguous_loop_edge_count,
      loop_closure_batch.unpaired_loop_edge_count,
      loop_closure_batch.failure_reason.empty() ? "none" :
      loop_closure_batch.failure_reason.c_str());

    for (std::size_t attempt_index = 0;
      attempt_index < loop_closure_batch.attempts.size(); ++attempt_index)
    {
      const auto & attempt = loop_closure_batch.attempts[attempt_index];
      const RoughnessLoopMemoryPrior memory_prior =
        attempt_index < loop_closure_memory_priors.size() ?
        loop_closure_memory_priors[attempt_index] :
        RoughnessLoopMemoryPrior{};
      std::ostringstream chain_state_ids;
      std::ostringstream chain_unique_ids;
      for (std::size_t index = 0;
        index < attempt.chain_state_ids.size(); ++index)
      {
        if (index > 0U) {
          chain_state_ids << ',';
          chain_unique_ids << ',';
        }
        chain_state_ids << attempt.chain_state_ids[index];
        chain_unique_ids << attempt.chain_unique_ids[index];
      }
      RCLCPP_INFO(
        get_logger(),
        "KARTO_LOOP_CANDIDATE stamp_ns=%lld mapper_generation=%llu "
        "candidate_index=%zu edge_type=loop_closure sensor=%s "
        "chain_state_ids=%s chain_unique_ids=%s chain_size=%zu "
        "chain_available=%s coarse_parsed=%s coarse_response=%.9g "
        "coarse_response_threshold=%.9g coarse_cov_xx_m2=%.9g "
        "coarse_cov_yy_m2=%.9g coarse_cov_threshold_m2=%.9g "
        "coarse_accepted=%s fine_parsed=%s fine_response=%.9g "
        "fine_response_threshold=%.9g fine_accepted=%s disposition=%s "
        "listener_sequence_complete=%s rejection_seen=%s begin_seen=%s "
        "end_seen=%s source_state_id=%d source_unique_id=%d "
        "target_state_id=%d target_unique_id=%d actual_edge_present=%s "
        "accepted_without_edge=%s event_edge_count_before_available=%s "
        "event_edge_count_before=%zu event_edge_count_after=%zu "
        "event_new_edge_count=%zu event_edge_index=%zu "
        "link_info_available=%s edge_pairing_ambiguous=%s "
        "relative_x_m=%.9g relative_y_m=%.9g relative_yaw_rad=%.9g "
        "odom_relative_x_m=%.9g odom_relative_y_m=%.9g "
        "odom_relative_yaw_rad=%.9g measurement_vs_odom_translation_m=%.9g "
        "measurement_vs_odom_yaw_rad=%.9g "
        "measurement_vs_odom_covariance_weighted_disagreement_sq=%.9g "
        "memory_available=%s memory_strength=%.9g "
        "adaptive_duration_s=%.9g adaptive_distance_m=%.9g "
        "place_memory_available=%s place_source_available_count=%zu "
        "place_source_chain_size=%zu place_source_coverage=%.9g "
        "place_source_peak_unique_id=%d place_source_peak_strength=%.9g "
        "place_actual_source_available=%s "
        "place_actual_source_strength=%.9g place_target_available=%s "
        "place_target_strength=%.9g place_combined_strength=%.9g "
        "place_source_peak_roughness_score=%.9g "
        "place_target_roughness_score=%.9g "
        "covariance_xx=%.9g "
        "covariance_xy=%.9g covariance_xtheta=%.9g covariance_yy=%.9g "
        "covariance_ytheta=%.9g covariance_thetatheta=%.9g "
        "pre_residual_x_m=%.9g pre_residual_y_m=%.9g "
        "pre_residual_yaw_rad=%.9g post_residual_x_m=%.9g "
        "post_residual_y_m=%.9g post_residual_yaw_rad=%.9g "
        "post_covariance_weighted_residual_sq=%.9g",
        static_cast<long long>(stamp_ns),
        static_cast<unsigned long long>(mapper_generation_.load()),
        attempt.candidate_index,
        attempt.sensor_name.c_str(),
        chain_state_ids.str().c_str(),
        chain_unique_ids.str().c_str(),
        attempt.chain_state_ids.size(),
        attempt.candidate_chain_available ? "true" : "false",
        attempt.coarse_message_parsed ? "true" : "false",
        attempt.coarse_response,
        attempt.coarse_response_threshold,
        attempt.coarse_covariance_xx_m2,
        attempt.coarse_covariance_yy_m2,
        attempt.coarse_covariance_threshold_m2,
        attempt.coarse_accepted ? "true" : "false",
        attempt.fine_message_parsed ? "true" : "false",
        attempt.fine_response,
        attempt.fine_response_threshold,
        attempt.fine_accepted ? "true" : "false",
        kartoLoopClosureDispositionName(attempt.disposition),
        attempt.listener_sequence_complete ? "true" : "false",
        attempt.rejection_callback_seen ? "true" : "false",
        attempt.begin_callback_seen ? "true" : "false",
        attempt.end_callback_seen ? "true" : "false",
        attempt.source_state_id,
        attempt.source_unique_id,
        attempt.target_state_id,
        attempt.target_unique_id,
        attempt.actual_edge_present ? "true" : "false",
        attempt.accepted_without_edge ? "true" : "false",
        attempt.event_edge_count_before_available ? "true" : "false",
        attempt.event_edge_count_before,
        attempt.event_edge_count_after,
        attempt.event_new_edge_count,
        attempt.event_edge_index,
        attempt.link_info_available ? "true" : "false",
        attempt.edge_pairing_ambiguous ? "true" : "false",
        attempt.relative_transform.GetX(),
        attempt.relative_transform.GetY(),
        attempt.relative_transform.GetHeading(),
        attempt.odometric_relative_transform.GetX(),
        attempt.odometric_relative_transform.GetY(),
        attempt.odometric_relative_transform.GetHeading(),
        attempt.measurement_vs_odometry_translation_m,
        attempt.measurement_vs_odometry_yaw_rad,
        attempt.measurement_vs_odometry_covariance_weighted_disagreement_sq,
        adaptive_memory_state.available ? "true" : "false",
        adaptive_memory_state.strength,
        adaptive_memory_state.effective_duration_s,
        adaptive_memory_state.effective_distance_m,
        memory_prior.available ? "true" : "false",
        memory_prior.source_available_count,
        memory_prior.source_chain_size,
        memory_prior.source_coverage,
        memory_prior.source_peak_unique_id,
        memory_prior.source_peak_strength,
        memory_prior.actual_source_available ? "true" : "false",
        memory_prior.actual_source_strength,
        memory_prior.target_available ? "true" : "false",
        memory_prior.target_strength,
        memory_prior.combined_strength,
        memory_prior.source_peak_roughness_score,
        memory_prior.target_roughness_score,
        attempt.factor_covariance(0, 0),
        attempt.factor_covariance(0, 1),
        attempt.factor_covariance(0, 2),
        attempt.factor_covariance(1, 1),
        attempt.factor_covariance(1, 2),
        attempt.factor_covariance(2, 2),
        attempt.pre_residual_x_m,
        attempt.pre_residual_y_m,
        attempt.pre_residual_yaw_rad,
        attempt.post_residual_x_m,
        attempt.post_residual_y_m,
        attempt.post_residual_yaw_rad,
        attempt.post_covariance_weighted_residual_sq);
    }
  }

  if (near_chain_input.enabled) {
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_NEAR_CHAIN_DIAG stamp_ns=%lld mapper_generation=%llu "
      "state_id=%d reason=%s active=%s rough_trigger=%s "
      "window_complete=%s rough_samples=%zu score=%.6f "
      "spatial_input_valid=%s spatial_revisit=%s spatial_zone_count=%zu "
      "spatial_nearest_distance_m=%.6f spatial_heading_delta_deg=%.6f "
      "spatial_source_rough_stamp_ns=%lld "
      "spatial_passage_input_valid=%s spatial_passage=%s "
      "spatial_passage_nearest_distance_m=%.6f "
      "spatial_passage_source_age_s=%.6f "
      "spatial_passage_source_rough_stamp_ns=%lld "
      "spatial_pose_source=%s pose_graph_anchor_mode=%s "
      "recovery_until_stamp_ns=%lld "
      "loop_suppression_until_stamp_ns=%lld "
      "loop_suppression_active=%s loop_cooldown=%s "
      "baseline_threshold=%.6f "
      "requested_threshold=%.6f applied_threshold=%.6f "
      "restored_threshold=%.6f restore_success=%s processed=%s "
      "loop_closing_baseline=%s loop_closing_applied=%s "
      "loop_closing_restored=%s loop_closing_restore_success=%s "
      "sequential_available=%s sequential_response=%.6f "
      "sequential_translation_m=%.6f "
      "sequential_yaw_deg=%.6f process_delta_x_m=%.6f "
      "process_delta_y_m=%.6f process_delta_translation_m=%.6f "
      "process_delta_yaw_deg=%.6f edges_before=%zu edges_after=%zu "
      "new_edges=%zu new_nonsequential_edges=%zu",
      static_cast<long long>(stamp_ns),
      static_cast<unsigned long long>(mapper_generation_.load()),
      processed_scan ? processed_scan->GetStateId() : -1,
      roughnessNearChainGateReasonName(near_chain_decision.reason),
      near_chain_decision.active ? "true" : "false",
      near_chain_decision.rough_trigger ? "true" : "false",
      window && window->complete ? "true" : "false",
      window ? window->rough_sample_count : 0U,
      window ? window->score : std::numeric_limits<double>::quiet_NaN(),
      spatial_revisit_decision.input_valid ? "true" : "false",
      spatial_revisit_decision.matched ? "true" : "false",
      spatial_zone_count,
      spatial_revisit_decision.nearest_distance_m,
      spatial_revisit_decision.heading_delta_rad * kRadiansToDegrees,
      static_cast<long long>(
        spatial_revisit_decision.source_rough_stamp_ns),
      spatial_passage_decision.input_valid ? "true" : "false",
      spatial_passage_decision.matched ? "true" : "false",
      spatial_passage_decision.nearest_distance_m,
      spatial_passage_decision.source_age_s,
      static_cast<long long>(
        spatial_passage_decision.source_rough_stamp_ns),
      spatial_pose_source,
      spatial_memory_use_pose_graph_anchors_ ? "true" : "false",
      static_cast<long long>(
        near_chain_decision.next_recovery_until_stamp_ns),
      static_cast<long long>(
        near_chain_decision.next_loop_closing_suppression_until_stamp_ns),
      near_chain_decision.suppress_loop_closing ? "true" : "false",
      near_chain_decision.loop_closing_cooldown ? "true" : "false",
      threshold_override.baseline(),
      near_chain_decision.requested_threshold,
      threshold_override.applied(),
      threshold_override.restored(),
      threshold_restore_succeeded ? "true" : "false",
      processed ? "true" : "false",
      loop_closing_suppression.baseline() ? "true" : "false",
      loop_closing_suppression.applied() ? "true" : "false",
      loop_closing_suppression.restored() ? "true" : "false",
      loop_closing_restore_succeeded ? "true" : "false",
      sequential_preview.available ? "true" : "false",
      sequential_preview.response,
      sequential_preview.translation_m,
      sequential_preview.yaw_rad * kRadiansToDegrees,
      process_delta_x_m,
      process_delta_y_m,
      process_delta_translation_m,
      process_delta_yaw_rad * kRadiansToDegrees,
      edge_count_before,
      edge_count_after,
      new_edge_count,
      new_nonsequential_edge_count);
  }

  return processed_scan;
}

void RoughnessAwareSlamToolbox::trackProcessedScanLocked(
  karto::LocalizedRangeScan * processed_scan,
  karto::Mapper * mapper)
{
  if (!processed_scan || !mapper || !mapper->getParamUseScanMatching()) {
    return;
  }
  const std::string sensor_name = processed_scan->GetSensorName().ToString();
  karto::LocalizedRangeScanVector & running_scans =
    preview_running_scans_[sensor_name];
  running_scans.push_back(processed_scan);
  const std::size_t maximum_size = static_cast<std::size_t>(
    std::max(mapper->getParamScanBufferSize(), 1));
  const double maximum_distance = mapper->getParamScanBufferMaximumScanDistance();
  while (running_scans.size() > 1) {
    const double squared_distance =
      running_scans.front()->GetSensorPose().GetPosition().SquaredDistance(
      running_scans.back()->GetSensorPose().GetPosition());
    if (
      running_scans.size() <= maximum_size &&
      squared_distance <=
      karto::math::Square(maximum_distance) - karto::KT_TOLERANCE)
    {
      break;
    }
    running_scans.erase(running_scans.begin());
  }
  preview_last_scans_[sensor_name] = processed_scan;
}

void RoughnessAwareSlamToolbox::updateRoughnessPlaceRecognitionLocked(
  const karto::LocalizedRangeScan * processed_scan,
  const std::int64_t stamp_ns,
  const std::optional<WindowDecision> & window)
{
  if (!enable_roughness_place_recognition_observer_ || !processed_scan) {
    return;
  }

  const karto::Pose2 & odometric = processed_scan->GetOdometricPose();
  const karto::Pose2 & corrected = processed_scan->GetCorrectedPose();
  RoughnessPlaceObservation observation;
  observation.stamp_ns = stamp_ns;
  observation.scan_state_id = processed_scan->GetStateId();
  observation.scan_unique_id = processed_scan->GetUniqueId();
  observation.odom_x_m = odometric.GetX();
  observation.odom_y_m = odometric.GetY();
  observation.odom_heading_rad = odometric.GetHeading();
  observation.corrected_x_m = corrected.GetX();
  observation.corrected_y_m = corrected.GetY();
  observation.corrected_heading_rad = corrected.GetHeading();
  observation.quality_valid =
    window && window->complete && window->sample_count > 0U &&
    std::isfinite(window->score);
  observation.rough =
    observation.quality_valid && window->sustained &&
    window->score >= roughness_score_gate_;
  if (window) {
    observation.features = normalizedRoughnessPlaceFeatures(
      window->score, window->peak_signal, roughness_thresholds_);
  } else {
    observation.features.fill(std::numeric_limits<double>::quiet_NaN());
  }

  std::lock_guard<std::mutex> place_lock(roughness_place_mutex_);
  RoughnessPlaceTrackerUpdate update =
    roughness_place_event_tracker_.update(observation);
  RoughnessPlaceStableScan chain_scan;
  chain_scan.mapper_generation = mapper_generation_.load();
  chain_scan.sensor_name = processed_scan->GetSensorName().ToString();
  chain_scan.stamp_ns = stamp_ns;
  chain_scan.state_id = processed_scan->GetStateId();
  chain_scan.unique_id = processed_scan->GetUniqueId();
  chain_scan.odom_x_m = odometric.GetX();
  chain_scan.odom_y_m = odometric.GetY();
  chain_scan.odom_yaw_rad = odometric.GetHeading();
  chain_scan.quality_valid = observation.quality_valid;
  chain_scan.strict_stable =
    observation.quality_valid && window && window->rough_sample_count == 0U &&
    !window->sustained && window->score < roughness_score_gate_;
  karto::Mapper * place_mapper = smapper_ ? smapper_->getMapper() : nullptr;
  if (roughness_place_enable_reference_chain_observer_) {
    evaluateRoughnessPlaceScanChainUpdateLocked(
      roughness_place_scan_chain_tracker_.observe(chain_scan, update.action),
      place_mapper);
  }
  if (update.action == RoughnessPlaceTrackerAction::kStarted) {
    ++roughness_place_event_started_;
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_PLACE_EVENT_START stamp_ns=%lld state_id=%d "
      "memory_events=%zu action=observe graph_action=none",
      static_cast<long long>(stamp_ns),
      observation.scan_state_id,
      roughness_place_events_.size());
    return;
  }
  if (update.action == RoughnessPlaceTrackerAction::kTimestampReset) {
    roughness_place_events_.clear();
    roughness_place_scan_chain_events_.clear();
    roughness_place_injected_passages_.clear();
    roughness_place_reference_chain_observed_passages_.clear();
    ++roughness_place_timestamp_reset_;
    RCLCPP_WARN(
      get_logger(),
      "ROUGHNESS_PLACE_RESET stamp_ns=%lld reason=timestamp_rollback",
      static_cast<long long>(stamp_ns));
    return;
  }
  if (!update.finalized_event) {
    return;
  }

  ++roughness_place_event_finalized_;
  RoughnessPlaceEvent event = std::move(*update.finalized_event);
  if (event.descriptor.valid()) {
    ++roughness_place_descriptor_valid_;
  }
  const std::vector<RoughnessPlaceCandidate> candidates =
    rankedRoughnessPlaceCandidates(
    event,
    roughness_place_events_,
    roughness_place_candidate_top_k_,
    roughness_place_min_revisit_separation_s_,
    roughness_place_min_state_separation_);
  if (roughness_place_enable_reference_chain_observer_) {
    evaluateRoughnessPlaceScanChainUpdateLocked(
      roughness_place_scan_chain_tracker_.finalizeEvent(
        event, candidates, chain_scan),
      place_mapper);
  }
  const RoughnessPlaceCandidate candidate = candidates.empty() ?
    RoughnessPlaceCandidate{} : candidates.front();
  double corrected_center_distance_m =
    std::numeric_limits<double>::quiet_NaN();
  double odom_center_distance_m =
    std::numeric_limits<double>::quiet_NaN();
  double corrected_post_heading_delta_rad =
    std::numeric_limits<double>::quiet_NaN();
  if (candidate.valid && candidate.event_index < roughness_place_events_.size()) {
    ++roughness_place_candidate_available_;
    const RoughnessPlaceEvent & reference =
      roughness_place_events_[candidate.event_index];
    const double event_center_x =
      0.5 * (event.pre_anchor.corrected_x_m + event.post_anchor.corrected_x_m);
    const double event_center_y =
      0.5 * (event.pre_anchor.corrected_y_m + event.post_anchor.corrected_y_m);
    const double reference_center_x = 0.5 *
      (reference.pre_anchor.corrected_x_m + reference.post_anchor.corrected_x_m);
    const double reference_center_y = 0.5 *
      (reference.pre_anchor.corrected_y_m + reference.post_anchor.corrected_y_m);
    corrected_center_distance_m = std::hypot(
      event_center_x - reference_center_x,
      event_center_y - reference_center_y);
    const double event_odom_center_x =
      0.5 * (event.pre_anchor.odom_x_m + event.post_anchor.odom_x_m);
    const double event_odom_center_y =
      0.5 * (event.pre_anchor.odom_y_m + event.post_anchor.odom_y_m);
    const double reference_odom_center_x = 0.5 *
      (reference.pre_anchor.odom_x_m + reference.post_anchor.odom_x_m);
    const double reference_odom_center_y = 0.5 *
      (reference.pre_anchor.odom_y_m + reference.post_anchor.odom_y_m);
    odom_center_distance_m = std::hypot(
      event_odom_center_x - reference_odom_center_x,
      event_odom_center_y - reference_odom_center_y);
    corrected_post_heading_delta_rad = normalizedAngle(
      event.post_anchor.corrected_heading_rad -
      reference.post_anchor.corrected_heading_rad);
  }

  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_PLACE_EVENT event_id=%llu pre_stamp_ns=%lld post_stamp_ns=%lld "
    "pre_state_id=%d post_state_id=%d pre_unique_id=%d post_unique_id=%d "
    "pre_odom_x_m=%.9g pre_odom_y_m=%.9g pre_odom_yaw_rad=%.9g "
    "post_odom_x_m=%.9g post_odom_y_m=%.9g post_odom_yaw_rad=%.9g "
    "rough_observations=%zu "
    "path_length_m=%.6f descriptor_bins=%zu descriptor_channels=%zu "
    "descriptor_valid=%s reference_events=%zu eligible_candidates=%zu "
    "candidate_available=%s "
    "candidate_event_id=%llu candidate_direction=%s candidate_score=%.6f "
    "candidate_shape_score=%.6f candidate_amplitude_score=%.6f "
    "candidate_length_score=%.6f candidate_second_score=%.6f "
    "candidate_margin=%.6f candidate_event_separation_s=%.6f "
    "candidate_minimum_state_separation=%zu odom_center_distance_m=%.6f "
    "corrected_center_distance_m=%.6f "
    "corrected_post_heading_delta_deg=%.6f action=observe graph_action=none",
    static_cast<unsigned long long>(event.event_id),
    static_cast<long long>(event.start_stamp_ns),
    static_cast<long long>(event.end_stamp_ns),
    event.pre_anchor_state_id,
    event.post_anchor_state_id,
    event.pre_anchor.scan_unique_id,
    event.post_anchor.scan_unique_id,
    event.pre_anchor.odom_x_m,
    event.pre_anchor.odom_y_m,
    event.pre_anchor.odom_heading_rad,
    event.post_anchor.odom_x_m,
    event.post_anchor.odom_y_m,
    event.post_anchor.odom_heading_rad,
    event.rough_observation_count,
    event.descriptor.path_length_m,
    event.descriptor.bin_count,
    event.descriptor.valid_channel_count,
    event.descriptor.valid() ? "true" : "false",
    roughness_place_events_.size(),
    candidates.size(),
    candidate.valid ? "true" : "false",
    static_cast<unsigned long long>(candidate.event_id),
    candidate.valid ?
    roughnessPlaceMatchDirectionName(candidate.match.direction) : "none",
    candidate.match.score,
    candidate.match.shape_score,
    candidate.match.amplitude_score,
    candidate.match.length_score,
    candidate.second_best_score,
    candidate.score_margin,
    candidate.event_separation_s,
    candidate.minimum_state_id_separation,
    odom_center_distance_m,
    corrected_center_distance_m,
    corrected_post_heading_delta_rad * kRadiansToDegrees);

  std::vector<RoughnessPlaceCandidateEvaluation> evaluations;
  std::vector<std::vector<RoughnessPlaceCandidateEvaluation>>
  direction_hypotheses;
  std::vector<RoughnessPlaceConstraintHypothesisSelection> direction_selections;
  std::vector<std::string> direction_selection_reasons;
  if (enable_roughness_place_scan_match_observer_) {
    karto::Mapper * mapper = smapper_ ? smapper_->getMapper() : nullptr;
    const karto::Name sensor_name = processed_scan->GetSensorName();
    evaluations.reserve(candidates.size());
    direction_hypotheses.reserve(candidates.size());
    direction_selections.reserve(candidates.size());
    direction_selection_reasons.reserve(candidates.size());
    for (const RoughnessPlaceCandidate & current_candidate : candidates) {
      ++roughness_place_scan_match_attempted_;
      const RoughnessPlaceEvent & reference =
        roughness_place_events_[current_candidate.event_index];
      const auto evaluate_direction =
        [this, mapper, &sensor_name, &reference, &event, &current_candidate](
        const RoughnessPlaceMatchDirection direction) {
          RoughnessPlaceCandidateEvaluation evaluation;
          evaluation.candidate = current_candidate;
          evaluation.descriptor_direction = current_candidate.match.direction;
          evaluation.candidate.match.direction = direction;
          evaluation.pairing = roughnessPlaceAnchorPairing(
            reference, event, direction);
          evaluation.reference_pre_scan = findGraphScan(
            mapper, sensor_name, evaluation.pairing.reference_pre_state_id);
          evaluation.reference_post_scan = findGraphScan(
            mapper, sensor_name, evaluation.pairing.reference_post_state_id);
          evaluation.query_for_reference_pre_scan = findGraphScan(
            mapper,
            sensor_name,
            evaluation.pairing.query_for_reference_pre_state_id);
          evaluation.query_for_reference_post_scan = findGraphScan(
            mapper,
            sensor_name,
            evaluation.pairing.query_for_reference_post_state_id);
          if (evaluation.pairing.valid) {
            evaluation.pre_anchor_match = observeRoughnessPlaceAnchorScanMatch(
              mapper,
              evaluation.reference_pre_scan,
              evaluation.query_for_reference_pre_scan);
            evaluation.post_anchor_match = observeRoughnessPlaceAnchorScanMatch(
              mapper,
              evaluation.reference_post_scan,
              evaluation.query_for_reference_post_scan);
          } else {
            evaluation.pre_anchor_match.failure_reason = "invalid_anchor_pairing";
            evaluation.post_anchor_match.failure_reason = "invalid_anchor_pairing";
          }
          if (
            evaluation.pre_anchor_match.forward.available &&
            evaluation.post_anchor_match.forward.available)
          {
            constexpr double kPiRad = 3.14159265358979323846;
            const double query_yaw_offset_rad =
              direction == RoughnessPlaceMatchDirection::kReverse ? kPiRad : 0.0;
            const RoughnessPlaceRelativeMotionConsistency consistency =
              roughnessPlaceRelativeMotionConsistency(
              roughnessPlacePose(
                evaluation.reference_pre_scan->GetCorrectedPose()),
              roughnessPlacePose(
                evaluation.reference_post_scan->GetCorrectedPose()),
              roughnessPlacePose(
                evaluation.pre_anchor_match.forward.matched_robot_pose,
                query_yaw_offset_rad),
              roughnessPlacePose(
                evaluation.post_anchor_match.forward.matched_robot_pose,
                query_yaw_offset_rad));
            if (consistency.available) {
              evaluation.forward_alignment_translation_error_m =
                consistency.translation_error_m;
              evaluation.forward_alignment_yaw_error_rad =
                consistency.yaw_error_rad;
            }
          }
          if (
            evaluation.pre_anchor_match.reverse.available &&
            evaluation.post_anchor_match.reverse.available)
          {
            constexpr double kPiRad = 3.14159265358979323846;
            const double query_yaw_offset_rad =
              direction == RoughnessPlaceMatchDirection::kReverse ? kPiRad : 0.0;
            const RoughnessPlaceRelativeMotionConsistency consistency =
              roughnessPlaceRelativeMotionConsistency(
              roughnessPlacePose(
                evaluation.pre_anchor_match.reverse.matched_robot_pose),
              roughnessPlacePose(
                evaluation.post_anchor_match.reverse.matched_robot_pose),
              roughnessPlacePose(
                evaluation.query_for_reference_pre_scan->GetCorrectedPose(),
                query_yaw_offset_rad),
              roughnessPlacePose(
                evaluation.query_for_reference_post_scan->GetCorrectedPose(),
                query_yaw_offset_rad));
            if (consistency.available) {
              evaluation.reverse_alignment_translation_error_m =
                consistency.translation_error_m;
              evaluation.reverse_alignment_yaw_error_rad =
                consistency.yaw_error_rad;
            }
          }
          evaluation.evidence = placePairEvidence(
            evaluation.pre_anchor_match,
            evaluation.post_anchor_match,
            evaluation.forward_alignment_translation_error_m,
            evaluation.forward_alignment_yaw_error_rad,
            evaluation.reverse_alignment_translation_error_m,
            evaluation.reverse_alignment_yaw_error_rad);
          evaluation.decision = evaluateRoughnessPlaceConstraint(
            evaluation.evidence, roughness_place_constraint_thresholds_);
          evaluation.quality = roughnessPlaceConstraintQuality(
            evaluation.evidence, roughness_place_constraint_thresholds_);
          return evaluation;
        };

      std::vector<RoughnessPlaceCandidateEvaluation> hypotheses;
      hypotheses.reserve(roughness_place_evaluate_both_directions_ ? 2U : 1U);
      hypotheses.push_back(evaluate_direction(current_candidate.match.direction));
      RoughnessPlaceConstraintHypothesisSelection direction_selection;
      std::string direction_selection_reason = "descriptor_only";
      if (roughness_place_evaluate_both_directions_) {
        const RoughnessPlaceMatchDirection alternate_direction =
          current_candidate.match.direction == RoughnessPlaceMatchDirection::kForward ?
          RoughnessPlaceMatchDirection::kReverse :
          RoughnessPlaceMatchDirection::kForward;
        hypotheses.push_back(evaluate_direction(alternate_direction));
        const RoughnessPlaceConstraintHypothesis preferred{
          hypotheses[0].candidate.match.direction,
          hypotheses[0].decision,
          hypotheses[0].quality};
        const RoughnessPlaceConstraintHypothesis alternate{
          hypotheses[1].candidate.match.direction,
          hypotheses[1].decision,
          hypotheses[1].quality};
        direction_selection = selectRoughnessPlaceConstraintHypothesis(
          preferred, alternate);
        direction_selection_reason =
          roughnessPlaceConstraintHypothesisSelectionReasonName(
          direction_selection.reason);
      } else {
        direction_selection.selected = true;
        direction_selection.fail_open = hypotheses[0].decision.fail_open;
        direction_selection.selected_index = 0U;
      }
      roughness_place_direction_hypotheses_ += hypotheses.size();

      RoughnessPlaceCandidateEvaluation evaluation = hypotheses.front();
      if (direction_selection.selected) {
        evaluation = hypotheses[direction_selection.selected_index];
        if (direction_selection.direction_overridden) {
          ++roughness_place_direction_overrides_;
        }
      } else if (direction_selection.fail_open) {
        evaluation.decision = RoughnessPlaceConstraintDecision{};
        ++roughness_place_direction_fail_open_;
      }
      if (evaluation.evidence.available && !direction_selection.fail_open) {
        ++roughness_place_scan_match_available_;
      } else {
        ++roughness_place_scan_match_fail_open_;
      }
      if (evaluation.decision.accepted) {
        ++roughness_place_quality_accepted_;
      } else if (evaluation.decision.fail_open) {
        ++roughness_place_quality_fail_open_;
      } else {
        ++roughness_place_quality_rejected_;
      }
      evaluations.push_back(std::move(evaluation));
      direction_hypotheses.push_back(std::move(hypotheses));
      direction_selections.push_back(direction_selection);
      direction_selection_reasons.push_back(std::move(direction_selection_reason));
    }
  }

  std::optional<std::size_t> selected_evaluation_index;
  for (std::size_t index = 0U; index < evaluations.size(); ++index) {
    if (evaluations[index].decision.accepted) {
      selected_evaluation_index = index;
      break;
    }
  }

  for (std::size_t candidate_index = 0U;
    candidate_index < evaluations.size(); ++candidate_index)
  {
    const RoughnessPlaceCandidateEvaluation & evaluation =
      evaluations[candidate_index];
    const RoughnessPlaceCandidate & evaluated_candidate = evaluation.candidate;
    const RoughnessPlaceConstraintHypothesisSelection & direction_selection =
      direction_selections[candidate_index];
    for (std::size_t hypothesis_index = 0U;
      hypothesis_index < direction_hypotheses[candidate_index].size();
      ++hypothesis_index)
    {
      const RoughnessPlaceCandidateEvaluation & hypothesis =
        direction_hypotheses[candidate_index][hypothesis_index];
      const bool direction_selected =
        !roughness_place_evaluate_both_directions_ ? hypothesis_index == 0U :
        direction_selection.selected &&
        direction_selection.selected_index == hypothesis_index;
      RCLCPP_INFO(
        get_logger(),
        "ROUGHNESS_PLACE_DIRECTION_HYPOTHESIS stamp_ns=%lld "
        "mapper_generation=%llu event_id=%llu candidate_event_id=%llu "
        "candidate_rank=%zu hypothesis_index=%zu descriptor_direction=%s "
        "evaluated_direction=%s preferred=%s quality_accepted=%s "
        "fail_open=%s reason=%s quality_available=%s quality_risk=%.9g "
        "minimum_fine_response=%.9g "
        "maximum_pair_alignment_translation_error_m=%.9g "
        "maximum_pair_alignment_yaw_error_rad=%.9g selected=%s "
        "direction_overridden=%s selection_fail_open=%s "
        "selection_reason=%s action=observe graph_action=none",
        static_cast<long long>(stamp_ns),
        static_cast<unsigned long long>(mapper_generation_.load()),
        static_cast<unsigned long long>(event.event_id),
        static_cast<unsigned long long>(hypothesis.candidate.event_id),
        candidate_index + 1U,
        hypothesis_index,
        roughnessPlaceMatchDirectionName(hypothesis.descriptor_direction),
        roughnessPlaceMatchDirectionName(hypothesis.candidate.match.direction),
        hypothesis_index == 0U ? "true" : "false",
        hypothesis.decision.accepted ? "true" : "false",
        hypothesis.decision.fail_open ? "true" : "false",
        roughnessPlaceConstraintDecisionReasonName(hypothesis.decision.reason),
        hypothesis.quality.available ? "true" : "false",
        hypothesis.quality.maximum_gate_utilization,
        hypothesis.quality.minimum_fine_response,
        hypothesis.decision.maximum_pair_alignment_translation_error_m,
        hypothesis.decision.maximum_pair_alignment_yaw_error_rad,
        direction_selected ? "true" : "false",
        direction_selected && hypothesis_index == 1U ? "true" : "false",
        direction_selection.fail_open ? "true" : "false",
        direction_selection_reasons[candidate_index].c_str());
    }
    const auto log_anchor_match =
      [this, &event, &evaluated_candidate, candidate_index](
      const char * anchor_role,
      const std::int32_t reference_state_id,
      const std::int32_t query_state_id,
      const RoughnessPlaceAnchorScanMatch & match) {
        const std::string failure_reason = logToken(match.failure_reason);
        const std::string forward_failure = logToken(
          match.forward.failure_reason);
        const std::string reverse_failure = logToken(
          match.reverse.failure_reason);
        RCLCPP_INFO(
          get_logger(),
          "ROUGHNESS_PLACE_ANCHOR_SCAN_MATCH event_id=%llu "
          "candidate_event_id=%llu candidate_rank=%zu candidate_direction=%s "
          "anchor_role=%s "
          "reference_state_id=%d query_state_id=%d state_id_separation=%d "
          "descriptor_score=%.9g descriptor_margin=%.9g "
          "available=%s fail_open=%s failure=%s "
          "forward_available=%s forward_failure=%s "
          "forward_coarse_response=%.9g forward_fine_response=%.9g "
          "forward_coarse_cov_xx=%.9g forward_coarse_cov_xy=%.9g "
          "forward_coarse_cov_xyaw=%.9g forward_coarse_cov_yy=%.9g "
          "forward_coarse_cov_yyaw=%.9g forward_coarse_cov_yawyaw=%.9g "
          "forward_fine_cov_xx=%.9g forward_fine_cov_xy=%.9g "
          "forward_fine_cov_xyaw=%.9g forward_fine_cov_yy=%.9g "
          "forward_fine_cov_yyaw=%.9g forward_fine_cov_yawyaw=%.9g "
          "forward_proposal_disagreement_translation_m=%.9g "
          "forward_proposal_disagreement_yaw_rad=%.9g "
          "reverse_available=%s reverse_failure=%s "
          "reverse_coarse_response=%.9g reverse_fine_response=%.9g "
          "reverse_coarse_cov_xx=%.9g reverse_coarse_cov_xy=%.9g "
          "reverse_coarse_cov_xyaw=%.9g reverse_coarse_cov_yy=%.9g "
          "reverse_coarse_cov_yyaw=%.9g reverse_coarse_cov_yawyaw=%.9g "
          "reverse_fine_cov_xx=%.9g reverse_fine_cov_xy=%.9g "
          "reverse_fine_cov_xyaw=%.9g reverse_fine_cov_yy=%.9g "
          "reverse_fine_cov_yyaw=%.9g reverse_fine_cov_yawyaw=%.9g "
          "reverse_proposal_disagreement_translation_m=%.9g "
          "reverse_proposal_disagreement_yaw_rad=%.9g "
          "odom_relative_x_m=%.9g odom_relative_y_m=%.9g "
          "odom_relative_yaw_rad=%.9g "
          "matched_relative_x_m=%.9g matched_relative_y_m=%.9g "
          "matched_relative_yaw_rad=%.9g "
          "measurement_vs_odometry_translation_m=%.9g "
          "measurement_vs_odometry_yaw_rad=%.9g "
          "bidirectional_translation_error_m=%.9g "
          "bidirectional_yaw_error_rad=%.9g "
          "directional_alignment_translation_error_m=%.9g "
          "directional_alignment_yaw_error_rad=%.9g "
          "branch_alignment_x_m=%.9g branch_alignment_y_m=%.9g "
          "branch_alignment_yaw_rad=%.9g action=observe graph_action=none",
          static_cast<unsigned long long>(event.event_id),
          static_cast<unsigned long long>(evaluated_candidate.event_id),
          candidate_index + 1U,
          roughnessPlaceMatchDirectionName(evaluated_candidate.match.direction),
          anchor_role,
          reference_state_id,
          query_state_id,
          std::abs(query_state_id - reference_state_id),
          evaluated_candidate.match.score,
          evaluated_candidate.score_margin,
          match.available ? "true" : "false",
          match.available ? "false" : "true",
          failure_reason.c_str(),
          match.forward.available ? "true" : "false",
          forward_failure.c_str(),
          match.forward.coarse_response,
          match.forward.fine_response,
          match.forward.coarse_covariance(0, 0),
          match.forward.coarse_covariance(0, 1),
          match.forward.coarse_covariance(0, 2),
          match.forward.coarse_covariance(1, 1),
          match.forward.coarse_covariance(1, 2),
          match.forward.coarse_covariance(2, 2),
          match.forward.fine_covariance(0, 0),
          match.forward.fine_covariance(0, 1),
          match.forward.fine_covariance(0, 2),
          match.forward.fine_covariance(1, 1),
          match.forward.fine_covariance(1, 2),
          match.forward.fine_covariance(2, 2),
          match.forward.proposal_disagreement_translation_m,
          match.forward.proposal_disagreement_yaw_rad,
          match.reverse.available ? "true" : "false",
          reverse_failure.c_str(),
          match.reverse.coarse_response,
          match.reverse.fine_response,
          match.reverse.coarse_covariance(0, 0),
          match.reverse.coarse_covariance(0, 1),
          match.reverse.coarse_covariance(0, 2),
          match.reverse.coarse_covariance(1, 1),
          match.reverse.coarse_covariance(1, 2),
          match.reverse.coarse_covariance(2, 2),
          match.reverse.fine_covariance(0, 0),
          match.reverse.fine_covariance(0, 1),
          match.reverse.fine_covariance(0, 2),
          match.reverse.fine_covariance(1, 1),
          match.reverse.fine_covariance(1, 2),
          match.reverse.fine_covariance(2, 2),
          match.reverse.proposal_disagreement_translation_m,
          match.reverse.proposal_disagreement_yaw_rad,
          match.odometric_relative_transform.GetX(),
          match.odometric_relative_transform.GetY(),
          match.odometric_relative_transform.GetHeading(),
          match.forward_relative_transform.GetX(),
          match.forward_relative_transform.GetY(),
          match.forward_relative_transform.GetHeading(),
          match.measurement_vs_odometry_translation_m,
          match.measurement_vs_odometry_yaw_rad,
          match.bidirectional_translation_error_m,
          match.bidirectional_yaw_error_rad,
          match.alignment_translation_error_m,
          match.alignment_yaw_error_rad,
          match.forward_branch_alignment.GetX(),
          match.forward_branch_alignment.GetY(),
          match.forward_branch_alignment.GetHeading());
      };
    log_anchor_match(
      "reference_pre",
      evaluation.pairing.reference_pre_state_id,
      evaluation.pairing.query_for_reference_pre_state_id,
      evaluation.pre_anchor_match);
    log_anchor_match(
      "reference_post",
      evaluation.pairing.reference_post_state_id,
      evaluation.pairing.query_for_reference_post_state_id,
      evaluation.post_anchor_match);
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_PLACE_PAIR_CONSISTENCY event_id=%llu "
      "candidate_event_id=%llu candidate_rank=%zu candidate_direction=%s "
      "attempted=true "
      "available=%s fail_open=%s reference_pre_state_id=%d "
      "query_for_reference_pre_state_id=%d reference_post_state_id=%d "
      "query_for_reference_post_state_id=%d "
      "method=relative_passage_motion_v1 "
      "forward_alignment_translation_error_m=%.9g "
      "forward_alignment_yaw_error_rad=%.9g "
      "reverse_alignment_translation_error_m=%.9g "
      "reverse_alignment_yaw_error_rad=%.9g action=observe graph_action=none",
      static_cast<unsigned long long>(event.event_id),
      static_cast<unsigned long long>(evaluated_candidate.event_id),
      candidate_index + 1U,
      roughnessPlaceMatchDirectionName(evaluated_candidate.match.direction),
      evaluation.evidence.available ? "true" : "false",
      evaluation.evidence.available ? "false" : "true",
      evaluation.pairing.reference_pre_state_id,
      evaluation.pairing.query_for_reference_pre_state_id,
      evaluation.pairing.reference_post_state_id,
      evaluation.pairing.query_for_reference_post_state_id,
      evaluation.forward_alignment_translation_error_m,
      evaluation.forward_alignment_yaw_error_rad,
      evaluation.reverse_alignment_translation_error_m,
      evaluation.reverse_alignment_yaw_error_rad);
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_PLACE_CANDIDATE_DECISION stamp_ns=%lld mapper_generation=%llu "
      "event_id=%llu "
      "candidate_event_id=%llu candidate_rank=%zu event_separation_s=%.9g "
      "minimum_state_id_separation=%zu descriptor_score=%.9g "
      "descriptor_margin=%.9g descriptor_direction=%s evaluated_direction=%s "
      "direction_overridden=%s direction_selection_fail_open=%s "
      "direction_selection_reason=%s quality_available=%s quality_risk=%.9g "
      "quality_minimum_fine_response=%.9g "
      "quality_accepted=%s fail_open=%s reason=%s "
      "selected=%s selected_anchor_role=%s selected_minimum_fine_response=%.9g "
      "maximum_pair_alignment_translation_error_m=%.9g "
      "maximum_pair_alignment_yaw_error_rad=%.9g "
      "reference_pre_state_id=%d reference_pre_unique_id=%d "
      "query_pre_state_id=%d query_pre_unique_id=%d "
      "reference_post_state_id=%d reference_post_unique_id=%d "
      "query_post_state_id=%d query_post_unique_id=%d "
      "action=%s graph_action=none",
      static_cast<long long>(stamp_ns),
      static_cast<unsigned long long>(mapper_generation_.load()),
      static_cast<unsigned long long>(event.event_id),
      static_cast<unsigned long long>(evaluated_candidate.event_id),
      candidate_index + 1U,
      evaluated_candidate.event_separation_s,
      evaluated_candidate.minimum_state_id_separation,
      evaluated_candidate.match.score,
      evaluated_candidate.score_margin,
      roughnessPlaceMatchDirectionName(evaluation.descriptor_direction),
      roughnessPlaceMatchDirectionName(evaluated_candidate.match.direction),
      direction_selection.direction_overridden ? "true" : "false",
      direction_selection.fail_open ? "true" : "false",
      direction_selection_reasons[candidate_index].c_str(),
      evaluation.quality.available ? "true" : "false",
      evaluation.quality.maximum_gate_utilization,
      evaluation.quality.minimum_fine_response,
      evaluation.decision.accepted ? "true" : "false",
      evaluation.decision.fail_open ? "true" : "false",
      roughnessPlaceConstraintDecisionReasonName(evaluation.decision.reason),
      selected_evaluation_index &&
      *selected_evaluation_index == candidate_index ? "true" : "false",
      evaluation.decision.accepted ?
      (evaluation.decision.selected_anchor_index == 0U ? "pre" : "post") :
      "none",
      evaluation.decision.selected_minimum_fine_response,
      evaluation.decision.maximum_pair_alignment_translation_error_m,
      evaluation.decision.maximum_pair_alignment_yaw_error_rad,
      evaluation.pairing.reference_pre_state_id,
      evaluation.reference_pre_scan ?
      evaluation.reference_pre_scan->GetUniqueId() : -1,
      evaluation.pairing.query_for_reference_pre_state_id,
      evaluation.query_for_reference_pre_scan ?
      evaluation.query_for_reference_pre_scan->GetUniqueId() : -1,
      evaluation.pairing.reference_post_state_id,
      evaluation.reference_post_scan ?
      evaluation.reference_post_scan->GetUniqueId() : -1,
      evaluation.pairing.query_for_reference_post_state_id,
      evaluation.query_for_reference_post_scan ?
      evaluation.query_for_reference_post_scan->GetUniqueId() : -1,
      roughness_place_constraint_mode_.c_str());
  }

  std::uint64_t constraint_candidate_event_id = 0U;
  std::size_t constraint_candidate_rank = 0U;
  const char * selected_anchor_role = "none";
  std::vector<RoughnessPlacePendingConstraint> pending_constraints;
  std::int32_t constraint_source_state_id = -1;
  std::int32_t constraint_target_state_id = -1;
  std::int32_t constraint_source_unique_id = -1;
  std::int32_t constraint_target_unique_id = -1;
  std::size_t constraint_edge_count_before = 0U;
  std::size_t constraint_edge_count_after = 0U;
  std::int64_t constraint_edge_index = -1;
  karto::Pose2 constraint_measurement = unavailablePose();
  karto::Matrix3 constraint_covariance;
  setMatrixUnavailable(constraint_covariance);
  karto::Matrix3 final_constraint_covariance;
  setMatrixUnavailable(final_constraint_covariance);
  RoughnessPlaceConstraintResidual pre_residual;
  RoughnessPlaceConstraintResidual post_residual;
  RoughnessPlaceConstraintPassage constraint_passage;
  bool correlated_passage_duplicate = false;
  std::uint64_t correlated_with_reference_event_id = 0U;
  std::uint64_t correlated_with_query_event_id = 0U;
  std::int64_t reference_intervening_state_count = -1;
  std::int64_t query_intervening_state_count = -1;
  bool constraint_injected = false;
  bool constraint_duplicate = false;
  bool constraint_group_partial = false;
  bool constraint_fail_open = true;
  std::string constraint_reason = "no_quality_accepted_candidate";
  std::string graph_action = "none";
  if (selected_evaluation_index) {
    RoughnessPlaceCandidateEvaluation & selected =
      evaluations[*selected_evaluation_index];
    constraint_candidate_event_id = selected.candidate.event_id;
    constraint_candidate_rank = *selected_evaluation_index + 1U;
    const bool use_validated_pair =
      roughness_place_constraint_anchor_strategy_ == "validated_pair";
    const auto append_constraint =
      [this, &selected, &pending_constraints](
      const char * anchor_role,
      karto::LocalizedRangeScan * source_scan,
      karto::LocalizedRangeScan * target_scan,
      const RoughnessPlaceAnchorScanMatch & anchor_match,
      const bool include_pair_consistency_variance,
      const double covariance_scale) {
        RoughnessPlacePendingConstraint constraint;
        constraint.anchor_role = anchor_role;
        constraint.source_scan = source_scan;
        constraint.target_scan = target_scan;
        constraint.anchor_match = &anchor_match;
        constraint.measurement = anchor_match.forward_relative_transform;
        constraint.covariance = makeRoughnessPlaceConstraintCovariance(
          anchor_match,
          selected.decision,
          roughness_place_constraint_pair_covariance_scale_,
          roughness_place_constraint_xy_variance_floor_m2_,
          roughness_place_constraint_yaw_variance_floor_rad2_,
          include_pair_consistency_variance,
          covariance_scale);
        constraint.pre_residual = roughnessPlaceConstraintResidual(
          source_scan,
          target_scan,
          constraint.measurement,
          constraint.covariance);
        pending_constraints.push_back(std::move(constraint));
      };
    if (use_validated_pair) {
      selected_anchor_role = "validated_pair";
      append_constraint(
        "pre",
        selected.reference_pre_scan,
        selected.query_for_reference_pre_scan,
        selected.pre_anchor_match,
        false,
        roughness_place_constraint_group_covariance_scale_);
      append_constraint(
        "post",
        selected.reference_post_scan,
        selected.query_for_reference_post_scan,
        selected.post_anchor_match,
        false,
        roughness_place_constraint_group_covariance_scale_);
    } else {
      const bool use_pre_anchor = selected.decision.selected_anchor_index == 0U;
      selected_anchor_role = use_pre_anchor ? "pre" : "post";
      append_constraint(
        selected_anchor_role,
        use_pre_anchor ? selected.reference_pre_scan :
        selected.reference_post_scan,
        use_pre_anchor ? selected.query_for_reference_pre_scan :
        selected.query_for_reference_post_scan,
        use_pre_anchor ? selected.pre_anchor_match : selected.post_anchor_match,
        true,
        1.0);
    }
    if (selected.candidate.event_index < roughness_place_events_.size()) {
      constraint_passage = roughnessPlaceConstraintPassage(
        roughness_place_events_[selected.candidate.event_index], event);
      for (const auto & injected_passage : roughness_place_injected_passages_) {
        const RoughnessPlacePassageCorrelation correlation =
          correlateRoughnessPlaceConstraintPassages(
          injected_passage, constraint_passage);
        if (!correlation.valid || !correlation.correlated) {
          continue;
        }
        correlated_passage_duplicate = true;
        correlated_with_reference_event_id =
          injected_passage.reference_event_id;
        correlated_with_query_event_id = injected_passage.query_event_id;
        reference_intervening_state_count = static_cast<std::int64_t>(
          correlation.reference_intervening_state_count);
        query_intervening_state_count = static_cast<std::int64_t>(
          correlation.query_intervening_state_count);
        break;
      }
    }
    if (roughness_place_constraint_mode_ == "observe") {
      constraint_reason = "observe_only";
      constraint_fail_open = false;
    } else {
      karto::Mapper * mapper = smapper_ ? smapper_->getMapper() : nullptr;
      auto * graph = mapper ? mapper->GetGraph() : nullptr;
      auto * solver = mapper ? mapper->getScanSolver() : nullptr;
      if (graph) {
        constraint_edge_count_before = graph->GetEdges().size();
        constraint_edge_count_after = constraint_edge_count_before;
      }
      bool invalid_backend_evidence =
        !mapper || !graph || !solver || pending_constraints.empty();
      for (std::size_t index = 0U;
        !invalid_backend_evidence && index < pending_constraints.size(); ++index)
      {
        const RoughnessPlacePendingConstraint & constraint =
          pending_constraints[index];
        invalid_backend_evidence =
          !constraint.source_scan || !constraint.target_scan ||
          constraint.source_scan == constraint.target_scan ||
          !constraint.anchor_match ||
          !finitePose(constraint.anchor_match->forward.matched_robot_pose) ||
          !finitePose(constraint.measurement) ||
          !finiteMatrix(constraint.covariance) ||
          constraint.covariance(0, 0) <= 0.0 ||
          constraint.covariance(1, 1) <= 0.0 ||
          constraint.covariance(2, 2) <= 0.0 ||
          !graphContainsScan(mapper, constraint.source_scan) ||
          !graphContainsScan(mapper, constraint.target_scan);
        for (std::size_t prior_index = 0U;
          !invalid_backend_evidence && prior_index < index; ++prior_index)
        {
          const RoughnessPlacePendingConstraint & prior =
            pending_constraints[prior_index];
          invalid_backend_evidence =
            (constraint.source_scan == prior.source_scan &&
            constraint.target_scan == prior.target_scan) ||
            (constraint.source_scan == prior.target_scan &&
            constraint.target_scan == prior.source_scan);
        }
      }
      if (invalid_backend_evidence) {
        ++roughness_place_constraint_failure_;
        constraint_reason = "missing_or_invalid_backend_evidence";
      } else if (correlated_passage_duplicate) {
        ++roughness_place_constraint_duplicate_;
        ++roughness_place_constraint_correlated_duplicate_;
        constraint_duplicate = true;
        constraint_fail_open = false;
        constraint_reason = "correlated_passage_preserved";
        graph_action = "correlated_passage_preserved";
      } else if (std::any_of(
          pending_constraints.begin(), pending_constraints.end(),
          [mapper](const RoughnessPlacePendingConstraint & constraint) {
            return graphAlreadyConnects(
              mapper, constraint.source_scan, constraint.target_scan);
          }))
      {
        ++roughness_place_constraint_duplicate_;
        constraint_duplicate = true;
        constraint_reason = "existing_group_edge_preserved";
      } else {
        bool insertion_succeeded = true;
        for (RoughnessPlacePendingConstraint & constraint : pending_constraints) {
          std::unique_ptr<karto::LinkInfo> link_info(new karto::LinkInfo(
              constraint.source_scan->GetCorrectedPose(),
              constraint.anchor_match->forward.matched_robot_pose,
              constraint.covariance));
          kt_bool is_new_edge = false;
          auto * edge = graph->AddEdge(
            constraint.source_scan, constraint.target_scan, is_new_edge);
          if (!edge || !is_new_edge) {
            insertion_succeeded = false;
            constraint_duplicate = edge && !is_new_edge;
            break;
          }
          edge->SetLabel(link_info.release());
          auto * final_link = dynamic_cast<karto::LinkInfo *>(edge->GetLabel());
          if (final_link) {
            constraint.final_covariance = final_link->GetCovariance();
          }
          solver->AddConstraint(edge);
          constraint.edge_index = static_cast<std::int64_t>(
            graph->GetEdges().size() - 1U);
          constraint.injected = true;
        }
        if (!insertion_succeeded) {
          ++roughness_place_constraint_failure_;
          constraint_group_partial = std::any_of(
            pending_constraints.begin(), pending_constraints.end(),
            [](const RoughnessPlacePendingConstraint & constraint) {
              return constraint.injected;
            });
          constraint_fail_open = !constraint_group_partial;
          constraint_reason = constraint_duplicate ?
            "existing_group_edge_preserved" :
            "constraint_group_insertion_failed_after_preflight";
          graph_action = constraint_group_partial ?
            "partial_group_inserted" : "none";
          if (constraint_duplicate) {
            ++roughness_place_constraint_duplicate_;
          }
        } else {
          mapper->CorrectPoses();
          constraint_edge_count_after = graph->GetEdges().size();
          constraint_injected = true;
          constraint_fail_open = false;
          constraint_reason = use_validated_pair ?
            "validated_pair_inserted" : "soft_edge_inserted";
          graph_action = use_validated_pair ?
            "validated_pair_inserted" : "soft_edge_inserted";
          ++roughness_place_constraint_injected_;
          roughness_place_injected_passages_.push_back(constraint_passage);
          if (
            roughness_place_injected_passages_.size() >
            roughness_place_max_events_)
          {
            roughness_place_injected_passages_.erase(
              roughness_place_injected_passages_.begin());
          }
          for (RoughnessPlacePendingConstraint & constraint :
            pending_constraints)
          {
            constraint.post_residual = roughnessPlaceConstraintResidual(
              constraint.source_scan,
              constraint.target_scan,
              constraint.measurement,
              constraint.covariance);
          }
        }
      }
    }
  }
  if (!pending_constraints.empty()) {
    const RoughnessPlacePendingConstraint & primary = pending_constraints.front();
    if (primary.source_scan) {
      constraint_source_state_id = primary.source_scan->GetStateId();
      constraint_source_unique_id = primary.source_scan->GetUniqueId();
    }
    if (primary.target_scan) {
      constraint_target_state_id = primary.target_scan->GetStateId();
      constraint_target_unique_id = primary.target_scan->GetUniqueId();
    }
    constraint_edge_index = primary.edge_index;
    constraint_measurement = primary.measurement;
    constraint_covariance = primary.covariance;
    final_constraint_covariance = primary.final_covariance;
    pre_residual = primary.pre_residual;
    post_residual = primary.post_residual;
  }
  const std::size_t injected_constraint_count = static_cast<std::size_t>(
    std::count_if(
      pending_constraints.begin(), pending_constraints.end(),
      [](const RoughnessPlacePendingConstraint & constraint) {
        return constraint.injected;
      }));
  for (std::size_t index = 0U; index < pending_constraints.size(); ++index) {
    const RoughnessPlacePendingConstraint & constraint =
      pending_constraints[index];
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_PLACE_CONSTRAINT_EDGE stamp_ns=%lld mapper_generation=%llu "
      "event_id=%llu candidate_event_id=%llu candidate_rank=%zu mode=%s "
      "anchor_strategy=%s group_size=%zu edge_group_index=%zu anchor_role=%s "
      "source_state_id=%d source_unique_id=%d target_state_id=%d "
      "target_unique_id=%d edge_index=%lld "
      "measurement_x_m=%.9g measurement_y_m=%.9g measurement_yaw_rad=%.9g "
      "covariance_xx=%.9g covariance_xy=%.9g covariance_xyaw=%.9g "
      "covariance_yy=%.9g covariance_yyaw=%.9g covariance_yawyaw=%.9g "
      "final_covariance_xx=%.9g final_covariance_xy=%.9g "
      "final_covariance_xyaw=%.9g final_covariance_yy=%.9g "
      "final_covariance_yyaw=%.9g final_covariance_yawyaw=%.9g "
      "pre_residual_x_m=%.9g pre_residual_y_m=%.9g "
      "pre_residual_translation_m=%.9g pre_residual_yaw_rad=%.9g "
      "pre_covariance_weighted_residual_sq=%.9g "
      "post_residual_x_m=%.9g post_residual_y_m=%.9g "
      "post_residual_translation_m=%.9g post_residual_yaw_rad=%.9g "
      "post_covariance_weighted_residual_sq=%.9g injected=%s "
      "group_partial=%s reason=%s action=%s graph_action=%s",
      static_cast<long long>(stamp_ns),
      static_cast<unsigned long long>(mapper_generation_.load()),
      static_cast<unsigned long long>(event.event_id),
      static_cast<unsigned long long>(constraint_candidate_event_id),
      constraint_candidate_rank,
      roughness_place_constraint_mode_.c_str(),
      roughness_place_constraint_anchor_strategy_.c_str(),
      pending_constraints.size(),
      index,
      constraint.anchor_role,
      constraint.source_scan ? constraint.source_scan->GetStateId() : -1,
      constraint.source_scan ? constraint.source_scan->GetUniqueId() : -1,
      constraint.target_scan ? constraint.target_scan->GetStateId() : -1,
      constraint.target_scan ? constraint.target_scan->GetUniqueId() : -1,
      static_cast<long long>(constraint.edge_index),
      constraint.measurement.GetX(),
      constraint.measurement.GetY(),
      constraint.measurement.GetHeading(),
      constraint.covariance(0, 0),
      constraint.covariance(0, 1),
      constraint.covariance(0, 2),
      constraint.covariance(1, 1),
      constraint.covariance(1, 2),
      constraint.covariance(2, 2),
      constraint.final_covariance(0, 0),
      constraint.final_covariance(0, 1),
      constraint.final_covariance(0, 2),
      constraint.final_covariance(1, 1),
      constraint.final_covariance(1, 2),
      constraint.final_covariance(2, 2),
      constraint.pre_residual.x_m,
      constraint.pre_residual.y_m,
      constraint.pre_residual.translation_m,
      constraint.pre_residual.yaw_rad,
      constraint.pre_residual.covariance_weighted_squared,
      constraint.post_residual.x_m,
      constraint.post_residual.y_m,
      constraint.post_residual.translation_m,
      constraint.post_residual.yaw_rad,
      constraint.post_residual.covariance_weighted_squared,
      constraint.injected ? "true" : "false",
      constraint_group_partial ? "true" : "false",
      constraint_reason.c_str(),
      roughness_place_constraint_mode_.c_str(),
      graph_action.c_str());
  }
  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_PLACE_CONSTRAINT stamp_ns=%lld mapper_generation=%llu "
    "event_id=%llu candidate_event_id=%llu "
    "candidate_rank=%zu mode=%s anchor_strategy=%s group_size=%zu "
    "injected_edge_count=%zu quality_accepted=%s fail_open=%s reason=%s "
    "selected_anchor_role=%s source_state_id=%d source_unique_id=%d "
    "target_state_id=%d target_unique_id=%d edge_type=roughness_place_soft "
    "edge_index=%lld edge_count_before=%zu edge_count_after=%zu "
    "measurement_x_m=%.9g measurement_y_m=%.9g measurement_yaw_rad=%.9g "
    "covariance_xx=%.9g covariance_yy=%.9g covariance_yawyaw=%.9g "
    "final_covariance_xx=%.9g final_covariance_xy=%.9g "
    "final_covariance_yy=%.9g final_covariance_yawyaw=%.9g "
    "pre_residual_x_m=%.9g pre_residual_y_m=%.9g "
    "pre_residual_translation_m=%.9g pre_residual_yaw_rad=%.9g "
    "pre_covariance_weighted_residual_sq=%.9g "
    "post_residual_x_m=%.9g post_residual_y_m=%.9g "
    "post_residual_translation_m=%.9g post_residual_yaw_rad=%.9g "
    "post_covariance_weighted_residual_sq=%.9g injected=%s duplicate=%s "
    "group_partial=%s pair_consistency_method=relative_passage_motion_v1 "
    "correlated_passage_duplicate=%s "
    "correlated_with_reference_event_id=%llu "
    "correlated_with_query_event_id=%llu "
    "reference_intervening_state_count=%lld "
    "query_intervening_state_count=%lld "
    "action=%s graph_action=%s",
    static_cast<long long>(stamp_ns),
    static_cast<unsigned long long>(mapper_generation_.load()),
    static_cast<unsigned long long>(event.event_id),
    static_cast<unsigned long long>(constraint_candidate_event_id),
    constraint_candidate_rank,
    roughness_place_constraint_mode_.c_str(),
    roughness_place_constraint_anchor_strategy_.c_str(),
    pending_constraints.size(),
    injected_constraint_count,
    selected_evaluation_index ? "true" : "false",
    constraint_fail_open ? "true" : "false",
    constraint_reason.c_str(),
    selected_anchor_role,
    constraint_source_state_id,
    constraint_source_unique_id,
    constraint_target_state_id,
    constraint_target_unique_id,
    static_cast<long long>(constraint_edge_index),
    constraint_edge_count_before,
    constraint_edge_count_after,
    constraint_measurement.GetX(),
    constraint_measurement.GetY(),
    constraint_measurement.GetHeading(),
    constraint_covariance(0, 0),
    constraint_covariance(1, 1),
    constraint_covariance(2, 2),
    final_constraint_covariance(0, 0),
    final_constraint_covariance(0, 1),
    final_constraint_covariance(1, 1),
    final_constraint_covariance(2, 2),
    pre_residual.x_m,
    pre_residual.y_m,
    pre_residual.translation_m,
    pre_residual.yaw_rad,
    pre_residual.covariance_weighted_squared,
    post_residual.x_m,
    post_residual.y_m,
    post_residual.translation_m,
    post_residual.yaw_rad,
    post_residual.covariance_weighted_squared,
    constraint_injected ? "true" : "false",
    constraint_duplicate ? "true" : "false",
    constraint_group_partial ? "true" : "false",
    correlated_passage_duplicate ? "true" : "false",
    static_cast<unsigned long long>(correlated_with_reference_event_id),
    static_cast<unsigned long long>(correlated_with_query_event_id),
    static_cast<long long>(reference_intervening_state_count),
    static_cast<long long>(query_intervening_state_count),
    roughness_place_constraint_mode_.c_str(),
    graph_action.c_str());

  roughness_place_events_.push_back(std::move(event));
  if (roughness_place_events_.size() > roughness_place_max_events_) {
    roughness_place_events_.erase(roughness_place_events_.begin());
  }
}

void RoughnessAwareSlamToolbox::evaluateRoughnessPlaceScanChainUpdateLocked(
  RoughnessPlaceScanChainUpdate update,
  karto::Mapper * mapper)
{
  if (!roughness_place_enable_reference_chain_observer_) {
    return;
  }
  for (RoughnessPlaceScanChainEvent & query_context : update.completed_events) {
    const auto evaluation_started_at = std::chrono::steady_clock::now();
    bool decision_evidence_available = false;
    bool decision_quality_accepted = false;
    bool decision_fail_open = true;
    bool decision_correlated_duplicate = false;
    std::string decision_reason = "missing_quality";
    std::uint64_t decision_candidate_event_id = 0U;
    std::size_t decision_candidate_rank = 0U;
    RoughnessPlaceMatchDirection decision_direction =
      RoughnessPlaceMatchDirection::kForward;
    double decision_quality_risk = std::numeric_limits<double>::quiet_NaN();
    double decision_minimum_fine_response =
      std::numeric_limits<double>::quiet_NaN();
    std::uint64_t correlated_with_reference_event_id = 0U;
    std::uint64_t correlated_with_query_event_id = 0U;
    std::int64_t reference_intervening_state_count = -1;
    std::int64_t query_intervening_state_count = -1;
    const auto log_decision = [
      this, &query_context, evaluation_started_at,
      &decision_evidence_available, &decision_quality_accepted,
      &decision_fail_open, &decision_correlated_duplicate, &decision_reason,
      &decision_candidate_event_id, &decision_candidate_rank,
      &decision_direction, &decision_quality_risk,
      &decision_minimum_fine_response,
      &correlated_with_reference_event_id, &correlated_with_query_event_id,
      &reference_intervening_state_count,
      &query_intervening_state_count]() {
        const double evaluation_duration_ms =
          std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - evaluation_started_at).count();
        std::ostringstream message;
        message << std::boolalpha
                << "ROUGHNESS_PLACE_REFERENCE_CHAIN_DECISION mapper_generation="
                << mapper_generation_.load()
                << " event_id=" << query_context.event.event_id
                << " available=" << decision_evidence_available
                << " fail_open=" << decision_fail_open
                << " quality_accepted=" << decision_quality_accepted
                << " reason=" << logToken(decision_reason)
                << " candidate_event_id=" << decision_candidate_event_id
                << " candidate_rank=" << decision_candidate_rank
                << " evaluated_direction="
                << (decision_candidate_rank > 0U ?
        roughnessPlaceMatchDirectionName(decision_direction) : "none")
                << " quality_risk=" << decision_quality_risk
                << " minimum_fine_response="
                << decision_minimum_fine_response
                << " correlated_passage_duplicate="
                << decision_correlated_duplicate
                << " correlated_with_reference_event_id="
                << correlated_with_reference_event_id
                << " correlated_with_query_event_id="
                << correlated_with_query_event_id
                << " reference_intervening_state_count="
                << reference_intervening_state_count
                << " query_intervening_state_count="
                << query_intervening_state_count
                << " evaluation_duration_ms=" << evaluation_duration_ms
                << " method=stable_side_reacquired_odom_clone_v3 "
                << "action=observe graph_action=none";
        RCLCPP_INFO(get_logger(), "%s", message.str().c_str());
      };
    const std::int64_t pre_chain_anchor_state_gap =
      query_context.pre_chain.empty() ? -1 :
      static_cast<std::int64_t>(query_context.event.pre_anchor_state_id) -
      static_cast<std::int64_t>(query_context.pre_chain.back().state_id);
    const double pre_chain_anchor_distance_m =
      query_context.pre_chain.empty() ?
      std::numeric_limits<double>::quiet_NaN() :
      std::hypot(
      query_context.pre_chain.back().odom_x_m -
      query_context.event.pre_anchor.odom_x_m,
      query_context.pre_chain.back().odom_y_m -
      query_context.event.pre_anchor.odom_y_m);
    const double pre_chain_anchor_time_gap_s =
      query_context.pre_chain.empty() ?
      std::numeric_limits<double>::quiet_NaN() :
      1.0e-9 * static_cast<double>(
      query_context.event.pre_anchor.stamp_ns -
      query_context.pre_chain.back().stamp_ns);
    const std::int64_t post_reacquisition_state_gap =
      query_context.post_chain.empty() ? -1 :
      static_cast<std::int64_t>(query_context.post_chain.front().state_id) -
      static_cast<std::int64_t>(query_context.event.post_anchor_state_id);
    const double post_reacquisition_distance_m =
      query_context.post_chain.empty() ?
      std::numeric_limits<double>::quiet_NaN() :
      std::hypot(
      query_context.post_chain.front().odom_x_m -
      query_context.event.post_anchor.odom_x_m,
      query_context.post_chain.front().odom_y_m -
      query_context.event.post_anchor.odom_y_m);
    const double post_reacquisition_time_gap_s =
      query_context.post_chain.empty() ?
      std::numeric_limits<double>::quiet_NaN() :
      1.0e-9 * static_cast<double>(
      query_context.post_chain.front().stamp_ns -
      query_context.event.post_anchor.stamp_ns);
    {
      std::ostringstream message;
      message << std::boolalpha
              << "ROUGHNESS_PLACE_REFERENCE_CHAIN_EVENT mapper_generation="
              << mapper_generation_.load()
              << " event_id=" << query_context.event.event_id
              << " available=" << query_context.available
              << " fail_open=" << !query_context.available
              << " failure=" << logToken(query_context.failure_reason)
              << " pre_anchor_state_id="
              << query_context.event.pre_anchor_state_id
              << " post_anchor_state_id="
              << query_context.event.post_anchor_state_id
              << " pre_chain_count=" << query_context.pre_chain.size()
              << " pre_chain_first_state_id="
              << (query_context.pre_chain.empty() ? -1 :
      query_context.pre_chain.front().state_id)
              << " pre_chain_last_state_id="
              << (query_context.pre_chain.empty() ? -1 :
      query_context.pre_chain.back().state_id)
              << " pre_chain_anchor_state_gap="
              << pre_chain_anchor_state_gap
              << " pre_chain_anchor_distance_m="
              << pre_chain_anchor_distance_m
              << " pre_chain_anchor_time_gap_s="
              << pre_chain_anchor_time_gap_s
              << " post_chain_count=" << query_context.post_chain.size()
              << " post_chain_first_state_id="
              << (query_context.post_chain.empty() ? -1 :
      query_context.post_chain.front().state_id)
              << " post_chain_last_state_id="
              << (query_context.post_chain.empty() ? -1 :
      query_context.post_chain.back().state_id)
              << " post_reacquisition_reset_count="
              << query_context.post_reacquisition_reset_count
              << " post_reacquisition_state_gap="
              << post_reacquisition_state_gap
              << " post_reacquisition_distance_m="
              << post_reacquisition_distance_m
              << " post_reacquisition_time_gap_s="
              << post_reacquisition_time_gap_s
              << " candidate_count=" << query_context.candidates.size()
              << " method=stable_side_reacquired_odom_clone_v3 "
              << "action=observe graph_action=none";
      RCLCPP_INFO(get_logger(), "%s", message.str().c_str());
    }
    if (!query_context.available || !mapper) {
      ++roughness_place_reference_chain_fail_open_;
      decision_reason = !query_context.available ?
        query_context.failure_reason : "missing_mapper";
      log_decision();
      continue;
    }

    struct ChainHypothesis
    {
      RoughnessPlaceCandidateEvaluation evaluation;
      RoughnessPlaceScanChainAnchorEvaluation pre_anchor;
      RoughnessPlaceScanChainAnchorEvaluation post_anchor;
      bool reference_context_available{false};
      std::string failure_reason{"not_evaluated"};
    };

    for (std::size_t candidate_index = 0U;
      candidate_index < query_context.candidates.size(); ++candidate_index)
    {
      const RoughnessPlaceCandidate & candidate =
        query_context.candidates[candidate_index];
      const auto reference_found = roughness_place_scan_chain_events_.find(
        candidate.event_id);
      const RoughnessPlaceScanChainEvent * reference_context =
        reference_found == roughness_place_scan_chain_events_.end() ? nullptr :
        &reference_found->second;
      std::vector<ChainHypothesis> hypotheses;
      hypotheses.reserve(2U);
      const RoughnessPlaceMatchDirection directions[2] = {
        candidate.match.direction,
        candidate.match.direction == RoughnessPlaceMatchDirection::kForward ?
        RoughnessPlaceMatchDirection::kReverse :
        RoughnessPlaceMatchDirection::kForward,
      };
      for (const RoughnessPlaceMatchDirection direction : directions) {
        ++roughness_place_reference_chain_attempted_;
        ChainHypothesis hypothesis;
        hypothesis.evaluation.candidate = candidate;
        hypothesis.evaluation.descriptor_direction = candidate.match.direction;
        hypothesis.evaluation.candidate.match.direction = direction;
        if (!reference_context || !reference_context->available) {
          hypothesis.failure_reason = "missing_reference_chain_context";
          ++roughness_place_reference_chain_fail_open_;
          hypotheses.push_back(std::move(hypothesis));
          continue;
        }
        hypothesis.reference_context_available = true;
        hypothesis.evaluation.pairing = roughnessPlaceScanChainPairing(
          *reference_context, query_context, direction);
        const auto & pairing = hypothesis.evaluation.pairing;
        const karto::Name sensor_name(query_context.pre_chain.front().sensor_name);
        hypothesis.evaluation.reference_pre_scan = findGraphScan(
          mapper, sensor_name, pairing.reference_pre_state_id);
        hypothesis.evaluation.reference_post_scan = findGraphScan(
          mapper, sensor_name, pairing.reference_post_state_id);
        hypothesis.evaluation.query_for_reference_pre_scan = findGraphScan(
          mapper, sensor_name, pairing.query_for_reference_pre_state_id);
        hypothesis.evaluation.query_for_reference_post_scan = findGraphScan(
          mapper, sensor_name, pairing.query_for_reference_post_state_id);
        hypothesis.pre_anchor = observeRoughnessPlaceScanChainAnchor(
          mapper,
          sensor_name,
          hypothesis.evaluation.reference_pre_scan,
          hypothesis.evaluation.query_for_reference_pre_scan,
          roughnessPlaceChainForAnchor(
            *reference_context, pairing.reference_pre_state_id),
          roughnessPlaceChainForAnchor(
            query_context, pairing.query_for_reference_pre_state_id),
          mapper_generation_.load());
        hypothesis.post_anchor = observeRoughnessPlaceScanChainAnchor(
          mapper,
          sensor_name,
          hypothesis.evaluation.reference_post_scan,
          hypothesis.evaluation.query_for_reference_post_scan,
          roughnessPlaceChainForAnchor(
            *reference_context, pairing.reference_post_state_id),
          roughnessPlaceChainForAnchor(
            query_context, pairing.query_for_reference_post_state_id),
          mapper_generation_.load());
        hypothesis.evaluation.pre_anchor_match = hypothesis.pre_anchor.match;
        hypothesis.evaluation.post_anchor_match = hypothesis.post_anchor.match;

        constexpr double kPiRad = 3.14159265358979323846;
        const double query_yaw_offset_rad =
          direction == RoughnessPlaceMatchDirection::kReverse ? kPiRad : 0.0;
        if (
          hypothesis.pre_anchor.match.forward.available &&
          hypothesis.post_anchor.match.forward.available)
        {
          const RoughnessPlaceRelativeMotionConsistency consistency =
            roughnessPlaceRelativeMotionConsistency(
            roughnessPlacePose(
              hypothesis.evaluation.reference_pre_scan->GetCorrectedPose()),
            roughnessPlacePose(
              hypothesis.evaluation.reference_post_scan->GetCorrectedPose()),
            roughnessPlacePose(
              hypothesis.pre_anchor.match.forward.matched_robot_pose,
              query_yaw_offset_rad),
            roughnessPlacePose(
              hypothesis.post_anchor.match.forward.matched_robot_pose,
              query_yaw_offset_rad));
          if (consistency.available) {
            hypothesis.evaluation.forward_alignment_translation_error_m =
              consistency.translation_error_m;
            hypothesis.evaluation.forward_alignment_yaw_error_rad =
              consistency.yaw_error_rad;
          }
        }
        if (
          hypothesis.pre_anchor.match.reverse.available &&
          hypothesis.post_anchor.match.reverse.available)
        {
          const RoughnessPlaceRelativeMotionConsistency consistency =
            roughnessPlaceRelativeMotionConsistency(
            roughnessPlacePose(
              hypothesis.pre_anchor.match.reverse.matched_robot_pose),
            roughnessPlacePose(
              hypothesis.post_anchor.match.reverse.matched_robot_pose),
            roughnessPlacePose(
              hypothesis.evaluation.query_for_reference_pre_scan->GetCorrectedPose(),
              query_yaw_offset_rad),
            roughnessPlacePose(
              hypothesis.evaluation.query_for_reference_post_scan->GetCorrectedPose(),
              query_yaw_offset_rad));
          if (consistency.available) {
            hypothesis.evaluation.reverse_alignment_translation_error_m =
              consistency.translation_error_m;
            hypothesis.evaluation.reverse_alignment_yaw_error_rad =
              consistency.yaw_error_rad;
          }
        }
        hypothesis.evaluation.evidence = placePairEvidence(
          hypothesis.evaluation.pre_anchor_match,
          hypothesis.evaluation.post_anchor_match,
          hypothesis.evaluation.forward_alignment_translation_error_m,
          hypothesis.evaluation.forward_alignment_yaw_error_rad,
          hypothesis.evaluation.reverse_alignment_translation_error_m,
          hypothesis.evaluation.reverse_alignment_yaw_error_rad);
        hypothesis.evaluation.decision = evaluateRoughnessPlaceConstraint(
          hypothesis.evaluation.evidence,
          roughness_place_constraint_thresholds_);
        hypothesis.evaluation.quality = roughnessPlaceConstraintQuality(
          hypothesis.evaluation.evidence,
          roughness_place_constraint_thresholds_);
        hypothesis.failure_reason =
          hypothesis.evaluation.evidence.available ? "none" :
          "pre=" + hypothesis.pre_anchor.failure_reason +
          ",post=" + hypothesis.post_anchor.failure_reason;
        if (hypothesis.evaluation.evidence.available) {
          decision_evidence_available = true;
          ++roughness_place_reference_chain_available_;
        } else {
          ++roughness_place_reference_chain_fail_open_;
        }
        hypotheses.push_back(std::move(hypothesis));
      }

      RoughnessPlaceConstraintHypothesisSelection selection;
      if (hypotheses.size() == 2U) {
        selection = selectRoughnessPlaceConstraintHypothesis(
          {
            hypotheses[0].evaluation.candidate.match.direction,
            hypotheses[0].evaluation.decision,
            hypotheses[0].evaluation.quality,
          },
          {
            hypotheses[1].evaluation.candidate.match.direction,
            hypotheses[1].evaluation.decision,
            hypotheses[1].evaluation.quality,
          });
      }
      const char * selection_reason =
        roughnessPlaceConstraintHypothesisSelectionReasonName(selection.reason);
      const auto log_anchor = [
        this, &query_context, &candidate, candidate_index](
        const ChainHypothesis & hypothesis,
        const std::size_t hypothesis_index,
        const char * anchor_role,
        const RoughnessPlaceScanChainAnchorEvaluation & anchor) {
          std::ostringstream message;
          message << std::boolalpha
                  << "ROUGHNESS_PLACE_REFERENCE_CHAIN_ANCHOR mapper_generation="
                  << mapper_generation_.load()
                  << " event_id=" << query_context.event.event_id
                  << " candidate_event_id=" << candidate.event_id
                  << " candidate_rank=" << candidate_index + 1U
                  << " hypothesis_index=" << hypothesis_index
                  << " descriptor_direction="
                  << roughnessPlaceMatchDirectionName(
            hypothesis.evaluation.descriptor_direction)
                  << " evaluated_direction="
                  << roughnessPlaceMatchDirectionName(
            hypothesis.evaluation.candidate.match.direction)
                  << " anchor_role=" << anchor_role
                  << " available=" << anchor.available
                  << " fail_open=" << !anchor.available
                  << " failure=" << logToken(anchor.failure_reason)
                  << " reference_base_scan_count="
                  << anchor.reference_base_scan_count
                  << " query_base_scan_count=" << anchor.query_base_scan_count
                  << " reference_first_state_id="
                  << anchor.reference_first_state_id
                  << " reference_last_state_id="
                  << anchor.reference_last_state_id
                  << " query_first_state_id=" << anchor.query_first_state_id
                  << " query_last_state_id=" << anchor.query_last_state_id
                  << " reference_maximum_anchor_distance_m="
                  << anchor.reference_maximum_anchor_distance_m
                  << " query_maximum_anchor_distance_m="
                  << anchor.query_maximum_anchor_distance_m
                  << " forward_coarse_response="
                  << anchor.match.forward.coarse_response
                  << " forward_fine_response="
                  << anchor.match.forward.fine_response
                  << " reverse_coarse_response="
                  << anchor.match.reverse.coarse_response
                  << " reverse_fine_response="
                  << anchor.match.reverse.fine_response
                  << " forward_fine_cov_xx="
                  << anchor.match.forward.fine_covariance(0, 0)
                  << " forward_fine_cov_yy="
                  << anchor.match.forward.fine_covariance(1, 1)
                  << " forward_fine_cov_yawyaw="
                  << anchor.match.forward.fine_covariance(2, 2)
                  << " reverse_fine_cov_xx="
                  << anchor.match.reverse.fine_covariance(0, 0)
                  << " reverse_fine_cov_yy="
                  << anchor.match.reverse.fine_covariance(1, 1)
                  << " reverse_fine_cov_yawyaw="
                  << anchor.match.reverse.fine_covariance(2, 2)
                  << " bidirectional_translation_error_m="
                  << anchor.match.bidirectional_translation_error_m
                  << " bidirectional_yaw_error_rad="
                  << anchor.match.bidirectional_yaw_error_rad
                  << " method=stable_side_reacquired_odom_clone_v3 action=observe "
                  << "graph_action=none";
          RCLCPP_INFO(get_logger(), "%s", message.str().c_str());
        };
      for (std::size_t index = 0U; index < hypotheses.size(); ++index) {
        const ChainHypothesis & hypothesis = hypotheses[index];
        log_anchor(hypothesis, index, "pre", hypothesis.pre_anchor);
        log_anchor(hypothesis, index, "post", hypothesis.post_anchor);
        const bool selected = selection.selected && selection.selected_index == index;
        std::ostringstream message;
        message << std::boolalpha
                << "ROUGHNESS_PLACE_REFERENCE_CHAIN_HYPOTHESIS mapper_generation="
                << mapper_generation_.load()
                << " event_id=" << query_context.event.event_id
                << " candidate_event_id=" << candidate.event_id
                << " candidate_rank=" << candidate_index + 1U
                << " hypothesis_index=" << index
                << " descriptor_direction="
                << roughnessPlaceMatchDirectionName(
          hypothesis.evaluation.descriptor_direction)
                << " evaluated_direction="
                << roughnessPlaceMatchDirectionName(
          hypothesis.evaluation.candidate.match.direction)
                << " reference_context_available="
                << hypothesis.reference_context_available
                << " available=" << hypothesis.evaluation.evidence.available
                << " fail_open=" << hypothesis.evaluation.decision.fail_open
                << " failure=" << logToken(hypothesis.failure_reason)
                << " quality_accepted="
                << hypothesis.evaluation.decision.accepted
                << " quality_reason="
                << roughnessPlaceConstraintDecisionReasonName(
          hypothesis.evaluation.decision.reason)
                << " quality_available="
                << hypothesis.evaluation.quality.available
                << " quality_risk="
                << hypothesis.evaluation.quality.maximum_gate_utilization
                << " minimum_fine_response="
                << hypothesis.evaluation.quality.minimum_fine_response
                << " forward_pair_translation_error_m="
                << hypothesis.evaluation.forward_alignment_translation_error_m
                << " forward_pair_yaw_error_rad="
                << hypothesis.evaluation.forward_alignment_yaw_error_rad
                << " reverse_pair_translation_error_m="
                << hypothesis.evaluation.reverse_alignment_translation_error_m
                << " reverse_pair_yaw_error_rad="
                << hypothesis.evaluation.reverse_alignment_yaw_error_rad
                << " selected=" << selected
                << " selection_fail_open=" << selection.fail_open
                << " selection_reason=" << selection_reason
                << " method=stable_side_reacquired_odom_clone_v3 action=observe "
                << "graph_action=none";
        RCLCPP_INFO(get_logger(), "%s", message.str().c_str());
      }

      if (
        !decision_quality_accepted && selection.selected &&
        selection.selected_index < hypotheses.size())
      {
        const ChainHypothesis & selected = hypotheses[selection.selected_index];
        if (selected.evaluation.decision.accepted) {
          decision_quality_accepted = true;
          decision_fail_open = false;
          decision_reason = "quality_accepted_unique_passage";
          decision_candidate_event_id = candidate.event_id;
          decision_candidate_rank = candidate_index + 1U;
          decision_direction = selected.evaluation.candidate.match.direction;
          decision_quality_risk =
            selected.evaluation.quality.maximum_gate_utilization;
          decision_minimum_fine_response =
            selected.evaluation.quality.minimum_fine_response;
          const RoughnessPlaceConstraintPassage passage =
            roughnessPlaceConstraintPassage(
            reference_context->event, query_context.event);
          if (!passage.valid()) {
            decision_fail_open = true;
            decision_reason = "accepted_passage_invalid";
          } else {
            for (const auto & observed_passage :
              roughness_place_reference_chain_observed_passages_)
            {
              const RoughnessPlacePassageCorrelation correlation =
                correlateRoughnessPlaceConstraintPassages(
                observed_passage, passage);
              if (!correlation.valid || !correlation.correlated) {
                continue;
              }
              decision_correlated_duplicate = true;
              decision_reason = "quality_accepted_correlated_duplicate";
              correlated_with_reference_event_id =
                observed_passage.reference_event_id;
              correlated_with_query_event_id = observed_passage.query_event_id;
              reference_intervening_state_count = static_cast<std::int64_t>(
                correlation.reference_intervening_state_count);
              query_intervening_state_count = static_cast<std::int64_t>(
                correlation.query_intervening_state_count);
              break;
            }
            if (!decision_correlated_duplicate) {
              roughness_place_reference_chain_observed_passages_.push_back(
                passage);
              if (
                roughness_place_reference_chain_observed_passages_.size() >
                roughness_place_max_events_)
              {
                roughness_place_reference_chain_observed_passages_.erase(
                  roughness_place_reference_chain_observed_passages_.begin());
              }
            }
          }
        }
      }
    }

    if (!decision_quality_accepted) {
      decision_fail_open = !decision_evidence_available;
      decision_reason = decision_evidence_available ?
        "quality_rejected" : "missing_quality";
    } else {
      ++roughness_place_reference_chain_event_accepted_;
      if (decision_correlated_duplicate) {
        ++roughness_place_reference_chain_correlated_duplicate_;
      }
    }
    log_decision();

    roughness_place_scan_chain_events_[query_context.event.event_id] =
      std::move(query_context);
    while (roughness_place_scan_chain_events_.size() > roughness_place_max_events_) {
      roughness_place_scan_chain_events_.erase(
        roughness_place_scan_chain_events_.begin());
    }
  }
}

std::optional<karto::Pose2>
RoughnessAwareSlamToolbox::predictPoseGraphSpatialPoseLocked(
  const karto::LocalizedRangeScan * range_scan) const
{
  if (!range_scan) {
    return std::nullopt;
  }
  const std::string sensor_name = range_scan->GetSensorName().ToString();
  const auto found = preview_last_scans_.find(sensor_name);
  if (found == preview_last_scans_.end() || !found->second) {
    return std::nullopt;
  }

  const karto::Pose2 & current_odom = range_scan->GetOdometricPose();
  const karto::Pose2 & last_odom = found->second->GetOdometricPose();
  const karto::Pose2 & last_corrected = found->second->GetCorrectedPose();
  if (
    !std::isfinite(current_odom.GetX()) ||
    !std::isfinite(current_odom.GetY()) ||
    !std::isfinite(current_odom.GetHeading()) ||
    !std::isfinite(last_odom.GetX()) ||
    !std::isfinite(last_odom.GetY()) ||
    !std::isfinite(last_odom.GetHeading()) ||
    !std::isfinite(last_corrected.GetX()) ||
    !std::isfinite(last_corrected.GetY()) ||
    !std::isfinite(last_corrected.GetHeading()))
  {
    return std::nullopt;
  }

  karto::Transform odom_to_corrected(last_odom, last_corrected);
  const karto::Pose2 predicted = odom_to_corrected.TransformPose(current_odom);
  if (
    !std::isfinite(predicted.GetX()) ||
    !std::isfinite(predicted.GetY()) ||
    !std::isfinite(predicted.GetHeading()))
  {
    return std::nullopt;
  }
  return predicted;
}

std::vector<RoughnessSpatialZone>
RoughnessAwareSlamToolbox::refreshPoseGraphSpatialAnchorsLocked(
  karto::Mapper * mapper)
{
  if (!mapper) {
    return {};
  }

  const karto::LocalizedRangeScanVector scans = mapper->GetAllProcessedScans();
  std::vector<RoughnessCorrectedScanPose> corrected_scan_poses;
  corrected_scan_poses.reserve(scans.size());
  for (const karto::LocalizedRangeScan * scan : scans) {
    if (!scan) {
      continue;
    }
    const karto::Pose2 & corrected = scan->GetCorrectedPose();
    corrected_scan_poses.push_back(
      {
        scan->GetUniqueId(),
        scan->GetStateId(),
        corrected.GetX(),
        corrected.GetY(),
        corrected.GetHeading(),
      });
  }

  auto result = refreshRoughnessPoseGraphAnchors(
    roughness_pose_graph_anchors_, corrected_scan_poses);
  ++spatial_memory_pose_graph_refresh_count_;
  spatial_memory_pose_graph_stale_remove_count_.fetch_add(
    result.missing_scan_count);
  spatial_memory_pose_graph_invalid_remove_count_.fetch_add(
    result.invalid_anchor_count + result.invalid_corrected_pose_count);
  if (
    result.missing_scan_count > 0 ||
    result.invalid_anchor_count > 0 ||
    result.invalid_corrected_pose_count > 0)
  {
    RCLCPP_WARN(
      get_logger(),
      "ROUGHNESS_POSE_GRAPH_ANCHOR_PRUNE missing_scan=%zu invalid_anchor=%zu "
      "invalid_corrected_pose=%zu retained=%zu",
      result.missing_scan_count,
      result.invalid_anchor_count,
      result.invalid_corrected_pose_count,
      roughness_pose_graph_anchors_.size());
  }
  return result.zones;
}

void RoughnessAwareSlamToolbox::rememberPoseGraphSpatialAnchorLocked(
  const karto::LocalizedRangeScan * processed_scan,
  const std::int64_t rough_stamp_ns)
{
  if (!processed_scan) {
    ++spatial_memory_invalid_pose_count_;
    return;
  }
  const karto::Pose2 & corrected = processed_scan->GetCorrectedPose();
  const RoughnessPoseGraphAnchor anchor{
    processed_scan->GetUniqueId(),
    processed_scan->GetStateId(),
    {
      corrected.GetX(),
      corrected.GetY(),
      corrected.GetHeading(),
      rough_stamp_ns,
      rough_stamp_ns,
    },
  };
  const auto result = rememberRoughnessPoseGraphAnchor(
    roughness_pose_graph_anchors_,
    anchor,
    spatial_memory_merge_radius_m_,
    spatial_memory_min_revisit_separation_s_,
    spatial_memory_max_zones_);
  if (!result.accepted) {
    ++spatial_memory_invalid_pose_count_;
    RCLCPP_WARN(
      get_logger(),
      "ROUGHNESS_POSE_GRAPH_ANCHOR_SKIP stamp_ns=%lld unique_id=%d state_id=%d "
      "reason=invalid_processed_scan_anchor",
      static_cast<long long>(rough_stamp_ns),
      processed_scan->GetUniqueId(),
      processed_scan->GetStateId());
    return;
  }
  if (result.merged) {
    ++spatial_memory_zone_merge_count_;
  }
  if (result.evicted) {
    ++spatial_memory_zone_evict_count_;
  }
  if (result.inserted) {
    ++spatial_memory_zone_insert_count_;
    ++spatial_memory_pose_graph_anchor_insert_count_;
  }
}

RoughnessAwareSlamToolbox::ScanMatchInnovation
RoughnessAwareSlamToolbox::previewScanMatchInnovation(
  karto::LaserRangeFinder * laser,
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
  karto::Pose2 & odom_pose,
  const FrozenScanContext * frozen_context,
  FrozenScanContext * captured_context)
{
  std::unique_ptr<karto::LocalizedRangeScan> range_scan(
    getLocalizedRangeScan(laser, scan, odom_pose));
  if (!range_scan) {
    return ScanMatchInnovation{};
  }

  boost::mutex::scoped_lock lock(smapper_mutex_);
  karto::Mapper * mapper = smapper_->getMapper();
  return previewLocalizedScanMatchInnovationLocked(
    range_scan.get(),
    mapper,
    frozen_context,
    captured_context);
}

RoughnessAwareSlamToolbox::ScanMatchInnovation
RoughnessAwareSlamToolbox::previewLocalizedScanMatchInnovationLocked(
  karto::LocalizedRangeScan * range_scan,
  karto::Mapper * mapper,
  const FrozenScanContext * frozen_context,
  FrozenScanContext * captured_context)
{
  ScanMatchInnovation innovation;
  if (!range_scan) {
    return innovation;
  }
  if (!mapper || !mapper->getParamUseScanMatching()) {
    return innovation;
  }
  karto::ScanMatcher * matcher = mapper->GetSequentialScanMatcher();
  if (!matcher) {
    return innovation;
  }
  const std::string sensor_name = range_scan->GetSensorName().ToString();
  karto::Pose2 last_odometric_pose;
  karto::Pose2 last_corrected_pose;
  karto::LocalizedRangeScanVector running_scans;
  if (frozen_context) {
    if (
      frozen_context->sensor_name != sensor_name ||
      frozen_context->scans.empty())
    {
      return innovation;
    }
    last_odometric_pose = frozen_context->last_odometric_pose;
    last_corrected_pose = frozen_context->last_corrected_pose;
    running_scans = frozen_context->scans;
  } else {
    const auto last_scan_found = preview_last_scans_.find(sensor_name);
    const auto running_scans_found = preview_running_scans_.find(sensor_name);
    if (
      last_scan_found == preview_last_scans_.end() ||
      running_scans_found == preview_running_scans_.end() ||
      !last_scan_found->second ||
      running_scans_found->second.empty())
    {
      return innovation;
    }
    karto::LocalizedRangeScan * last_scan = last_scan_found->second;
    last_odometric_pose = last_scan->GetOdometricPose();
    last_corrected_pose = last_scan->GetCorrectedPose();
    running_scans = running_scans_found->second;

    if (captured_context) {
      *captured_context = FrozenScanContext{};
      captured_context->sensor_name = sensor_name;
      captured_context->last_odometric_pose = last_odometric_pose;
      captured_context->last_corrected_pose = last_corrected_pose;
      const std::size_t first_index =
        running_scans.size() > post_confirmation_max_base_scans_ ?
        running_scans.size() - post_confirmation_max_base_scans_ : 0;
      try {
        for (std::size_t index = first_index; index < running_scans.size(); ++index) {
          karto::LocalizedRangeScan * source = running_scans[index];
          if (!source) {
            continue;
          }
          auto clone = std::make_unique<karto::LocalizedRangeScan>(
            source->GetSensorName(),
            source->GetRangeReadingsVector());
          clone->SetTime(source->GetTime());
          clone->SetOdometricPose(source->GetOdometricPose());
          clone->SetCorrectedPose(source->GetCorrectedPose());
          captured_context->source_state_ids.push_back(source->GetStateId());
          captured_context->scans.push_back(clone.get());
          captured_context->scan_storage.push_back(std::move(clone));
        }
      } catch (const std::exception & ex) {
        RCLCPP_WARN(
          get_logger(),
          "Failed to freeze pre-event scan context: %s",
          ex.what());
        *captured_context = FrozenScanContext{};
      }
      if (!captured_context->scans.empty()) {
        running_scans = captured_context->scans;
      }
    }
  }

  karto::Transform last_transform(last_odometric_pose, last_corrected_pose);
  range_scan->SetCorrectedPose(
    last_transform.TransformPose(range_scan->GetOdometricPose()));
  const karto::Pose2 predicted_pose = range_scan->GetCorrectedPose();
  innovation.predicted_x_m = predicted_pose.GetX();
  innovation.predicted_y_m = predicted_pose.GetY();
  innovation.predicted_yaw_rad = predicted_pose.GetHeading();

  karto::Pose2 best_sensor_pose;
  karto::Matrix3 covariance;
  covariance.SetToIdentity();
  try {
    innovation.response = matcher->MatchScan(
      range_scan,
      running_scans,
      best_sensor_pose,
      covariance);
  } catch (const karto::Exception & ex) {
    RCLCPP_WARN(
      get_logger(),
      "Roughness innovation preview failed: %s",
      ex.GetErrorMessage().c_str());
    return innovation;
  } catch (const std::exception & ex) {
    RCLCPP_WARN(
      get_logger(),
      "Roughness innovation preview failed: %s",
      ex.what());
    return innovation;
  }
  range_scan->SetSensorPose(best_sensor_pose);
  const karto::Pose2 matched_pose = range_scan->GetCorrectedPose();
  innovation.matched_x_m = matched_pose.GetX();
  innovation.matched_y_m = matched_pose.GetY();
  innovation.matched_yaw_rad = matched_pose.GetHeading();
  innovation.dx_m = matched_pose.GetX() - predicted_pose.GetX();
  innovation.dy_m = matched_pose.GetY() - predicted_pose.GetY();
  innovation.translation_m = std::hypot(innovation.dx_m, innovation.dy_m);
  innovation.yaw_rad = normalizedAngle(
    matched_pose.GetHeading() - predicted_pose.GetHeading());
  karto::Transform odom_correction(
    range_scan->GetOdometricPose(),
    matched_pose);
  const karto::Pose2 odom_correction_pose =
    odom_correction.TransformPose(karto::Pose2());
  innovation.odom_correction_x_m = odom_correction_pose.GetX();
  innovation.odom_correction_y_m = odom_correction_pose.GetY();
  innovation.odom_correction_yaw_rad = odom_correction_pose.GetHeading();
  innovation.covariance_xx = covariance(0, 0);
  innovation.covariance_xy = 0.5 * (covariance(0, 1) + covariance(1, 0));
  innovation.covariance_yy = covariance(1, 1);
  innovation.covariance_yawyaw = covariance(2, 2);

  const double covariance_xy_trace =
    innovation.covariance_xx + innovation.covariance_yy;
  const double covariance_xy_discriminant = std::sqrt(
    std::max(
      0.0,
      (innovation.covariance_xx - innovation.covariance_yy) *
      (innovation.covariance_xx - innovation.covariance_yy) +
      4.0 * innovation.covariance_xy * innovation.covariance_xy));
  const double covariance_xy_max_eigenvalue =
    0.5 * (covariance_xy_trace + covariance_xy_discriminant);
  const double covariance_xy_min_eigenvalue =
    0.5 * (covariance_xy_trace - covariance_xy_discriminant);
  if (std::isfinite(covariance_xy_max_eigenvalue)) {
    innovation.covariance_xy_max_eigenvalue = covariance_xy_max_eigenvalue;
  }
  if (
    std::isfinite(covariance_xy_min_eigenvalue) &&
    covariance_xy_min_eigenvalue > 1.0e-12 &&
    covariance_xy_max_eigenvalue >= covariance_xy_min_eigenvalue)
  {
    innovation.covariance_xy_condition =
      covariance_xy_max_eigenvalue / covariance_xy_min_eigenvalue;
  }

  karto::Matrix3 regularized_covariance = covariance;
  regularized_covariance(0, 0) += 1.0e-9;
  regularized_covariance(1, 1) += 1.0e-9;
  regularized_covariance(2, 2) += 1.0e-9;
  karto::Matrix3 information;
  if (regularized_covariance.InverseFast(information, 1.0e-14)) {
    const double innovation_vector[3] = {
      innovation.dx_m,
      innovation.dy_m,
      innovation.yaw_rad,
    };
    double covariance_weighted_disagreement_sq = 0.0;
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t column = 0; column < 3; ++column) {
        covariance_weighted_disagreement_sq +=
          innovation_vector[row] *
          information(row, column) *
          innovation_vector[column];
      }
    }
    if (
      std::isfinite(covariance_weighted_disagreement_sq) &&
      covariance_weighted_disagreement_sq >= 0.0)
    {
      innovation.scan_match_covariance_weighted_disagreement_sq =
        covariance_weighted_disagreement_sq;
    }
  }
  innovation.available =
    std::isfinite(innovation.response) &&
    std::isfinite(innovation.dx_m) &&
    std::isfinite(innovation.dy_m) &&
    std::isfinite(innovation.translation_m) &&
    std::isfinite(innovation.yaw_rad) &&
    std::isfinite(innovation.odom_correction_x_m) &&
    std::isfinite(innovation.odom_correction_y_m) &&
    std::isfinite(innovation.odom_correction_yaw_rad);
  return innovation;
}

bool RoughnessAwareSlamToolbox::armPostConfirmation(
  const std::int64_t stamp_ns,
  const double roughness_score,
  const RoughnessInnovationDecision & gate,
  const ScanMatchInnovation & innovation,
  FrozenScanContext context)
{
  if (
    context.scans.size() < post_confirmation_min_base_scans_ ||
    context.scans.size() != context.scan_storage.size())
  {
    ++post_confirmation_unavailable_;
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_POST_CONFIRMATION_SKIP rough_stamp_ns=%lld "
      "reason=insufficient_frozen_base frozen_base_scans=%zu required=%zu",
      static_cast<long long>(stamp_ns),
      context.scans.size(),
      post_confirmation_min_base_scans_);
    return false;
  }

  std::lock_guard<std::mutex> lock(post_confirmation_mutex_);
  if (pending_post_confirmation_) {
    ++post_confirmation_overlap_;
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_POST_CONFIRMATION_SKIP rough_stamp_ns=%lld "
      "reason=overlapping_event pending_rough_stamp_ns=%lld",
      static_cast<long long>(stamp_ns),
      static_cast<long long>(pending_post_confirmation_->stamp_ns));
    return false;
  }

  PostConfirmationCandidate candidate;
  candidate.stamp_ns = stamp_ns;
  candidate.roughness_score = roughness_score;
  candidate.translation_large = gate.translation_large;
  candidate.yaw_large = gate.yaw_large;
  candidate.innovation = innovation;
  candidate.context = std::move(context);
  pending_post_confirmation_.emplace(std::move(candidate));
  ++post_confirmation_armed_;
  return true;
}

void RoughnessAwareSlamToolbox::evaluatePendingPostConfirmation(
  karto::LaserRangeFinder * laser,
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
  karto::Pose2 & odom_pose,
  const std::optional<WindowDecision> & window)
{
  if (!enable_post_confirmation_) {
    return;
  }

  const std::int64_t post_stamp_ns = messageStampNs(scan->header.stamp);
  const bool stable =
    window &&
    window->complete &&
    !window->sustained &&
    window->rough_sample_count == 0 &&
    std::isfinite(window->score) &&
    window->score < roughness_score_gate_;

  std::optional<PostConfirmationCandidate> candidate;
  std::int64_t expired_rough_stamp_ns = 0;
  const char * expiration_reason = nullptr;
  {
    std::lock_guard<std::mutex> lock(post_confirmation_mutex_);
    if (!pending_post_confirmation_) {
      return;
    }
    const double age_s =
      static_cast<double>(
      post_stamp_ns - pending_post_confirmation_->stamp_ns) * 1.0e-9;
    if (age_s <= 0.0) {
      expired_rough_stamp_ns = pending_post_confirmation_->stamp_ns;
      pending_post_confirmation_.reset();
      expiration_reason = "scan_timestamp_rollback";
    } else if (age_s > post_confirmation_max_age_s_) {
      expired_rough_stamp_ns = pending_post_confirmation_->stamp_ns;
      pending_post_confirmation_.reset();
      expiration_reason = "stable_scan_timeout";
    } else if (stable) {
      candidate.emplace(std::move(*pending_post_confirmation_));
      pending_post_confirmation_.reset();
    }
  }

  if (expiration_reason) {
    ++post_confirmation_expired_;
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_POST_CONFIRMATION_SKIP rough_stamp_ns=%lld "
      "post_stamp_ns=%lld reason=%s",
      static_cast<long long>(expired_rough_stamp_ns),
      static_cast<long long>(post_stamp_ns),
      expiration_reason);
    return;
  }
  if (!candidate) {
    return;
  }

  const ScanMatchInnovation post_innovation =
    previewScanMatchInnovation(
    laser,
    scan,
    odom_pose,
    &candidate->context,
    nullptr);
  if (
    !post_innovation.available ||
    post_innovation.response < post_confirmation_min_response_)
  {
    ++post_confirmation_unavailable_;
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_POST_CONFIRMATION_SKIP rough_stamp_ns=%lld "
      "post_stamp_ns=%lld reason=%s response=%.6f frozen_base_scans=%zu",
      static_cast<long long>(candidate->stamp_ns),
      static_cast<long long>(post_stamp_ns),
      post_innovation.available ? "low_post_response" : "post_match_unavailable",
      post_innovation.response,
      candidate->context.scans.size());
    return;
  }

  const karto::Pose2 rough_transform(
    candidate->innovation.odom_correction_x_m,
    candidate->innovation.odom_correction_y_m,
    candidate->innovation.odom_correction_yaw_rad);
  const karto::Pose2 post_transform(
    post_innovation.odom_correction_x_m,
    post_innovation.odom_correction_y_m,
    post_innovation.odom_correction_yaw_rad);
  karto::Transform transform_disagreement(rough_transform, post_transform);
  const karto::Pose2 transform_disagreement_pose =
    transform_disagreement.TransformPose(karto::Pose2());
  const double transform_disagreement_translation_m = std::hypot(
    transform_disagreement_pose.GetX(),
    transform_disagreement_pose.GetY());
  const double transform_disagreement_yaw_rad =
    normalizedAngle(transform_disagreement_pose.GetHeading());
  const RoughnessPostConfirmationDecision decision =
    roughnessPostConfirmationDecision(
    candidate->translation_large,
    candidate->yaw_large,
    candidate->innovation.response,
    candidate->innovation.covariance_xy_max_eigenvalue,
    candidate->innovation.covariance_yawyaw,
    post_innovation.response,
    post_innovation.covariance_xy_max_eigenvalue,
    post_innovation.covariance_yawyaw,
    transform_disagreement_translation_m,
    transform_disagreement_yaw_rad,
    post_confirmation_translation_disagreement_gate_m_,
    post_confirmation_yaw_disagreement_gate_rad_,
    post_confirmation_min_response_gain_,
    post_confirmation_min_covariance_improvement_ratio_);

  ++post_confirmation_evaluated_;
  if (decision.confirmed_transient) {
    ++post_confirmation_confirmed_transient_;
  }
  const std::int32_t first_state_id =
    candidate->context.source_state_ids.empty() ?
    -1 : candidate->context.source_state_ids.front();
  const std::int32_t last_state_id =
    candidate->context.source_state_ids.empty() ?
    -1 : candidate->context.source_state_ids.back();
  const double age_s =
    static_cast<double>(post_stamp_ns - candidate->stamp_ns) * 1.0e-9;
  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_POST_CONFIRMATION rough_stamp_ns=%lld post_stamp_ns=%lld "
    "age_s=%.6f frozen_base_scans=%zu first_state_id=%d last_state_id=%d "
    "rough_score=%.6f rough_response=%.6f post_response=%.6f "
    "response_gain=%.6f rough_dx_m=%.6f rough_dy_m=%.6f rough_yaw_deg=%.6f "
    "post_dx_m=%.6f post_dy_m=%.6f post_yaw_deg=%.6f "
    "rough_odom_correction_x_m=%.6f rough_odom_correction_y_m=%.6f "
    "rough_odom_correction_yaw_deg=%.6f "
    "post_odom_correction_x_m=%.6f post_odom_correction_y_m=%.6f "
    "post_odom_correction_yaw_deg=%.6f "
    "transform_disagreement_translation_m=%.6f "
    "transform_disagreement_yaw_deg=%.6f "
    "xy_covariance_improvement_ratio=%.9g "
    "yaw_covariance_improvement_ratio=%.9g "
    "translation_inconsistent=%s yaw_inconsistent=%s "
    "confirmed_transient=%s action=observe",
    static_cast<long long>(candidate->stamp_ns),
    static_cast<long long>(post_stamp_ns),
    age_s,
    candidate->context.scans.size(),
    first_state_id,
    last_state_id,
    candidate->roughness_score,
    candidate->innovation.response,
    post_innovation.response,
    decision.response_gain,
    candidate->innovation.dx_m,
    candidate->innovation.dy_m,
    candidate->innovation.yaw_rad * kRadiansToDegrees,
    post_innovation.dx_m,
    post_innovation.dy_m,
    post_innovation.yaw_rad * kRadiansToDegrees,
    candidate->innovation.odom_correction_x_m,
    candidate->innovation.odom_correction_y_m,
    candidate->innovation.odom_correction_yaw_rad * kRadiansToDegrees,
    post_innovation.odom_correction_x_m,
    post_innovation.odom_correction_y_m,
    post_innovation.odom_correction_yaw_rad * kRadiansToDegrees,
    transform_disagreement_translation_m,
    transform_disagreement_yaw_rad * kRadiansToDegrees,
    decision.translation_covariance_improvement_ratio,
    decision.yaw_covariance_improvement_ratio,
    decision.translation_inconsistent ? "true" : "false",
    decision.yaw_inconsistent ? "true" : "false",
    decision.confirmed_transient ? "true" : "false");
}

void RoughnessAwareSlamToolbox::clearPostConfirmation(const char * reason)
{
  std::int64_t rough_stamp_ns = 0;
  {
    std::lock_guard<std::mutex> lock(post_confirmation_mutex_);
    if (!pending_post_confirmation_) {
      return;
    }
    rough_stamp_ns = pending_post_confirmation_->stamp_ns;
    pending_post_confirmation_.reset();
  }
  ++post_confirmation_expired_;
  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_POST_CONFIRMATION_SKIP rough_stamp_ns=%lld reason=%s",
    static_cast<long long>(rough_stamp_ns),
    reason);
}

void RoughnessAwareSlamToolbox::rememberRoughZoneLocked(
  const RoughnessSpatialZone & zone)
{
  constexpr long double kNanosecondsPerSecond = 1000000000.0L;
  const long double same_pass_limit_ns =
    static_cast<long double>(spatial_memory_min_revisit_separation_s_) *
    kNanosecondsPerSecond;
  for (auto & existing : roughness_spatial_zones_) {
    if (
      zone.rough_stamp_ns < existing.rough_stamp_ns ||
      static_cast<long double>(
        zone.rough_stamp_ns - existing.rough_stamp_ns) >=
      same_pass_limit_ns)
    {
      continue;
    }
    if (
      std::hypot(zone.x_m - existing.x_m, zone.y_m - existing.y_m) >
      spatial_memory_merge_radius_m_)
    {
      continue;
    }
    existing.last_rough_stamp_ns = std::max(
      existing.last_rough_stamp_ns > 0 ?
      existing.last_rough_stamp_ns : existing.rough_stamp_ns,
      zone.last_rough_stamp_ns > 0 ?
      zone.last_rough_stamp_ns : zone.rough_stamp_ns);
    ++spatial_memory_zone_merge_count_;
    return;
  }

  if (roughness_spatial_zones_.size() >= spatial_memory_max_zones_) {
    const auto oldest = std::min_element(
      roughness_spatial_zones_.begin(),
      roughness_spatial_zones_.end(),
      [](const RoughnessSpatialZone & lhs, const RoughnessSpatialZone & rhs) {
        const std::int64_t lhs_latest =
        lhs.last_rough_stamp_ns > 0 ?
        lhs.last_rough_stamp_ns : lhs.rough_stamp_ns;
        const std::int64_t rhs_latest =
        rhs.last_rough_stamp_ns > 0 ?
        rhs.last_rough_stamp_ns : rhs.rough_stamp_ns;
        return lhs_latest < rhs_latest;
      });
    roughness_spatial_zones_.erase(oldest);
    ++spatial_memory_zone_evict_count_;
  }
  roughness_spatial_zones_.push_back(zone);
  ++spatial_memory_zone_insert_count_;
}

RoughnessAwareSlamToolbox::ScanConditioningResult
RoughnessAwareSlamToolbox::conditionScan(
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
  const std::optional<WindowDecision> & window)
{
  ScanConditioningResult result;
  result.scan = scan;
  if (!enable_scan_conditioning_ || !scan) {
    return result;
  }

  const std::int64_t stamp_ns = messageStampNs(scan->header.stamp);
  const bool valid_stamp = isValidMessageTimestamp(
    scan->header.stamp.sec, scan->header.stamp.nanosec);
  const double stamp_s = valid_stamp ?
    static_cast<double>(stamp_ns) * 1.0e-9 :
    std::numeric_limits<double>::quiet_NaN();
  double fault_clock_s = get_clock()->now().seconds();
  if (!std::isfinite(fault_clock_s) || fault_clock_s <= 0.0) {
    fault_clock_s = stamp_s;
  }
  const LaserScanQuality input_quality = laserScanQuality(*scan);
  result.input_valid_beams = input_quality.valid_beam_count;
  result.output_valid_beams = input_quality.valid_beam_count;

  std::lock_guard<std::mutex> lock(scan_conditioning_mutex_);
  if (
    scan_conditioning_last_scan_stamp_ns_ > 0 &&
    stamp_ns < scan_conditioning_last_scan_stamp_ns_)
  {
    resetScanConditioningStateLocked();
    RCLCPP_WARN(
      get_logger(),
      "ROUGHNESS_SCAN_CONDITION_RESET reason=scan_timestamp_rollback "
      "stamp_ns=%lld",
      static_cast<long long>(stamp_ns));
  }
  if (valid_stamp) {
    scan_conditioning_last_scan_stamp_ns_ = stamp_ns;
  }

  if (window) {
    result.nearest_imu_dt_s = window->nearest_imu_dt_s;
    if (scan_conditioning_use_nearest_imu_) {
      result.score = window->nearest_imu_score;
      result.imu_available =
        window->nearest_imu_available &&
        window->nearest_imu_quality_valid &&
        std::isfinite(window->nearest_imu_dt_s) &&
        window->nearest_imu_dt_s <= scan_conditioning_max_scan_imu_dt_s_ &&
        std::isfinite(window->nearest_imu_score);
    } else {
      result.score = window->score;
      result.imu_available = window->complete && std::isfinite(window->score);
    }
  }

  const bool score_rough =
    result.imu_available &&
    result.score >= roughness_score_gate_;
  const bool score_hard =
    result.imu_available &&
    result.score >= scan_conditioning_hard_score_gate_;
  if (score_rough && valid_stamp) {
    scan_conditioning_last_rough_stamp_ns_ = stamp_ns;
  }
  if (score_hard && valid_stamp) {
    scan_conditioning_last_hard_stamp_ns_ = stamp_ns;
  }

  const auto within_hold = [stamp_ns](
    const std::int64_t event_stamp_ns,
    const double duration_s) {
      if (event_stamp_ns <= 0 || stamp_ns < event_stamp_ns || duration_s <= 0.0) {
        return false;
      }
      const long double age_ns =
        static_cast<long double>(stamp_ns) -
        static_cast<long double>(event_stamp_ns);
      return age_ns <=
             static_cast<long double>(duration_s) * 1000000000.0L;
    };
  result.rough_active = score_rough || within_hold(
    scan_conditioning_last_rough_stamp_ns_,
    scan_conditioning_hold_duration_s_);
  result.hard_active = score_hard || within_hold(
    scan_conditioning_last_hard_stamp_ns_,
    scan_conditioning_hard_hold_duration_s_);

  if (result.rough_active && valid_stamp) {
    if (scan_conditioning_event_start_stamp_ns_ <= 0) {
      scan_conditioning_event_start_stamp_ns_ = stamp_ns;
    }
    if (scan_conditioning_drop_start_stamp_ns_ <= 0) {
      scan_conditioning_drop_start_stamp_ns_ = stamp_ns;
    }
  } else if (!result.rough_active) {
    scan_conditioning_event_start_stamp_ns_ = 0;
    scan_conditioning_last_publish_stamp_ns_ = 0;
    scan_conditioning_drop_start_stamp_ns_ = 0;
  }

  bool used_fallback = false;
  auto apply_fallback = [this, &result, &used_fallback](
    const std::string & fallback_mode,
    const std::string & reason) {
      used_fallback = true;
      result.drop = fallback_mode == "drop";
      result.action = reason + (result.drop ? "_drop" : "_observe");
    };
  auto finish = [this, &result, &used_fallback, stamp_ns]() {
      if (result.drop) {
        ++scan_conditioning_drop_count_;
      } else {
        ++scan_conditioning_allow_count_;
      }
      if (used_fallback) {
        ++scan_conditioning_fallback_count_;
      }
      RCLCPP_INFO(
        get_logger(),
        "ROUGHNESS_SCAN_CONDITION stamp_ns=%lld mode=%s action=%s "
        "score=%.6f rough_active=%s hard_active=%s imu_available=%s "
        "nearest_imu_dt_s=%.6f input_valid_beams=%zu "
        "output_valid_beams=%zu overlap_beams=%zu removed_beams=%zu "
        "recovery_active=%s recovery_fault=%s",
        static_cast<long long>(stamp_ns),
        scanConditioningModeName(scan_conditioning_mode_),
        result.action.c_str(),
        result.score,
        result.rough_active ? "true" : "false",
        result.hard_active ? "true" : "false",
        result.imu_available ? "true" : "false",
        result.nearest_imu_dt_s,
        result.input_valid_beams,
        result.output_valid_beams,
        result.overlap_beams,
        result.removed_beams,
        result.recovery_active ? "true" : "false",
        result.recovery_fault ? "true" : "false");
      return result;
    };

  switch (scan_conditioning_mode_) {
    case ScanConditioningMode::kObserve:
      result.action = result.rough_active ? "rough_observe" : "allow";
      return finish();

    case ScanConditioningMode::kDrop:
      result.drop = result.rough_active;
      result.action = result.drop ? "rough_drop" : "allow";
      return finish();

    case ScanConditioningMode::kThrottle:
      if (!result.rough_active) {
        result.action = "allow";
        return finish();
      }
      if (
        valid_stamp &&
        scan_conditioning_max_continuous_drop_duration_s_ > 0.0 &&
        scan_conditioning_drop_start_stamp_ns_ > 0 &&
        static_cast<long double>(
          stamp_ns - scan_conditioning_drop_start_stamp_ns_) >=
        static_cast<long double>(
          scan_conditioning_max_continuous_drop_duration_s_) * 1000000000.0L)
      {
        scan_conditioning_last_publish_stamp_ns_ = stamp_ns;
        scan_conditioning_drop_start_stamp_ns_ = stamp_ns;
        result.action = "bridge_allow";
        ++scan_conditioning_bridge_count_;
        return finish();
      }
      if (result.hard_active && scan_conditioning_throttle_drop_hard_) {
        result.drop = true;
        result.action = "hard_drop";
        return finish();
      }
      if (
        valid_stamp &&
        (scan_conditioning_last_publish_stamp_ns_ <= 0 ||
        stamp_ns < scan_conditioning_last_publish_stamp_ns_ ||
        static_cast<long double>(
          stamp_ns - scan_conditioning_last_publish_stamp_ns_) >=
        static_cast<long double>(
          scan_conditioning_throttle_min_publish_period_s_) * 1000000000.0L))
      {
        scan_conditioning_last_publish_stamp_ns_ = stamp_ns;
        scan_conditioning_drop_start_stamp_ns_ = stamp_ns;
        result.action = "throttle_allow";
        return finish();
      }
      result.drop = true;
      result.action = "throttle_drop";
      return finish();

    case ScanConditioningMode::kQuarantine:
      {
        const bool recovery_qualified =
          result.imu_available &&
          !result.rough_active &&
          input_quality.valid_ratio >=
          scan_conditioning_quarantine_min_valid_ratio_;
        const bool imu_fault = scan_conditioning_imu_fail_open_gate_.update(
          fault_clock_s, result.imu_available);
        const RoughnessRecoveryDecision recovery =
          scan_conditioning_recovery_gate_.update(
          stamp_s,
          result.rough_active,
          recovery_qualified,
          fault_clock_s);
        result.recovery_active = recovery.quarantined;
        result.recovery_fault = recovery.fault_active || imu_fault;
        if (shouldFailClosedForInvalidScanTimestamp(
            true, true, valid_stamp))
        {
          result.drop = true;
          result.action = "quarantine_invalid_timestamp";
          ++scan_conditioning_quarantine_count_;
          return finish();
        }
        if (shouldFailOpenForUnavailableImu(
            true,
            true,
            result.imu_available,
            scan_conditioning_quarantine_fail_open_on_imu_fault_,
            imu_fault,
            valid_stamp,
            input_quality.valid_ratio >=
            scan_conditioning_quarantine_min_valid_ratio_))
        {
          result.action = "imu_unavailable_fail_open";
          return finish();
        }
        if (recovery.allow) {
          result.action = recovery.recovered ? "recovered_allow" : "allow";
          return finish();
        }
        result.drop = true;
        result.action = result.rough_active ?
          "quarantine_rough" : "quarantine_recovery";
        ++scan_conditioning_quarantine_count_;
        return finish();
      }

    case ScanConditioningMode::kStabilize:
      if (!result.rough_active) {
        if (result.imu_available) {
          rememberStableConditioningScan(*scan, stamp_ns, input_quality);
        }
        result.action = "allow";
        return finish();
      }
      if (!valid_stamp || !stable_conditioning_scan_) {
        apply_fallback(
          scan_conditioning_stabilize_fallback_mode_,
          valid_stamp ? "stabilize_no_reference" :
          "stabilize_invalid_timestamp");
        return finish();
      }
      {
        const double source_age_s = static_cast<double>(
          stamp_ns - stable_conditioning_scan_stamp_ns_) * 1.0e-9;
        const double event_duration_s = static_cast<double>(
          stamp_ns - scan_conditioning_event_start_stamp_ns_) * 1.0e-9;
        if (source_age_s < 0.0) {
          resetScanConditioningStateLocked();
          apply_fallback(
            scan_conditioning_stabilize_fallback_mode_,
            "stabilize_clock_reset");
          return finish();
        }
        if (
          scan_conditioning_stabilize_max_source_age_s_ > 0.0 &&
          source_age_s > scan_conditioning_stabilize_max_source_age_s_)
        {
          apply_fallback(
            scan_conditioning_stabilize_fallback_mode_,
            "stabilize_source_too_old");
          return finish();
        }
        if (
          event_duration_s < 0.0 ||
          (scan_conditioning_stabilize_max_event_duration_s_ > 0.0 &&
          event_duration_s > scan_conditioning_stabilize_max_event_duration_s_))
        {
          apply_fallback(
            scan_conditioning_stabilize_fallback_mode_,
            "stabilize_event_too_long");
          return finish();
        }
        if (
          scan_conditioning_last_publish_stamp_ns_ > 0 &&
          stamp_ns >= scan_conditioning_last_publish_stamp_ns_ &&
          static_cast<long double>(
            stamp_ns - scan_conditioning_last_publish_stamp_ns_) <
          static_cast<long double>(
            scan_conditioning_stabilize_min_publish_period_s_) *
          1000000000.0L)
        {
          apply_fallback(
            scan_conditioning_stabilize_fallback_mode_,
            "stabilize_rate_limited");
          return finish();
        }
        const auto transform = stableScanTransform(
          *scan, *stable_conditioning_scan_);
        if (!transform) {
          ++scan_conditioning_tf_failure_count_;
          apply_fallback(
            scan_conditioning_stabilize_fallback_mode_,
            "stabilize_tf_failed");
          return finish();
        }
        const RelativeMotion motion = relativeMotion(*transform);
        if (
          !motion.finite ||
          motion.translation_m > scan_conditioning_stabilize_max_translation_m_ ||
          std::fabs(motion.yaw_rad) > scan_conditioning_stabilize_max_yaw_rad_ ||
          motion.roll_pitch_rad >
          scan_conditioning_stabilize_max_roll_pitch_rad_)
        {
          apply_fallback(
            scan_conditioning_stabilize_fallback_mode_,
            "stabilize_motion_limit");
          return finish();
        }
        const std::size_t ratio_minimum = static_cast<std::size_t>(std::ceil(
            scan_conditioning_stabilize_min_output_valid_ratio_ *
            static_cast<double>(scan->ranges.size())));
        const std::size_t minimum_valid_beams = std::max(
          scan_conditioning_stabilize_min_output_valid_beams_,
          ratio_minimum);
        auto stabilized = warpLaserScan(
          *stable_conditioning_scan_,
          *scan,
          *transform,
          minimum_valid_beams);
        if (!stabilized) {
          apply_fallback(
            scan_conditioning_stabilize_fallback_mode_,
            "stabilize_sparse_output");
          return finish();
        }
        scan_conditioning_last_publish_stamp_ns_ = stamp_ns;
        result.output_valid_beams = stabilized->valid_beam_count;
        result.scan = std::make_shared<sensor_msgs::msg::LaserScan>(
          std::move(stabilized->scan));
        result.action = "stabilized_allow";
        ++scan_conditioning_stabilized_count_;
        return finish();
      }

    case ScanConditioningMode::kResidualFilter:
      if (!result.rough_active) {
        if (result.imu_available) {
          rememberStableConditioningScan(*scan, stamp_ns, input_quality);
        }
        result.action = "allow";
        return finish();
      }
      if (!valid_stamp || !stable_conditioning_scan_) {
        apply_fallback(
          scan_conditioning_residual_fallback_mode_,
          valid_stamp ? "residual_no_reference" :
          "residual_invalid_timestamp");
        return finish();
      }
      {
        const double source_age_s = static_cast<double>(
          stamp_ns - stable_conditioning_scan_stamp_ns_) * 1.0e-9;
        if (source_age_s < 0.0) {
          resetScanConditioningStateLocked();
          apply_fallback(
            scan_conditioning_residual_fallback_mode_,
            "residual_clock_reset");
          return finish();
        }
        if (
          scan_conditioning_residual_max_source_age_s_ > 0.0 &&
          source_age_s > scan_conditioning_residual_max_source_age_s_)
        {
          apply_fallback(
            scan_conditioning_residual_fallback_mode_,
            "residual_source_too_old");
          return finish();
        }
        const auto transform = stableScanTransform(
          *scan, *stable_conditioning_scan_);
        if (!transform) {
          ++scan_conditioning_tf_failure_count_;
          apply_fallback(
            scan_conditioning_residual_fallback_mode_,
            "residual_tf_failed");
          return finish();
        }
        const RelativeMotion motion = relativeMotion(*transform);
        if (
          !motion.finite ||
          motion.translation_m > scan_conditioning_residual_max_translation_m_ ||
          std::fabs(motion.yaw_rad) > scan_conditioning_residual_max_yaw_rad_ ||
          motion.roll_pitch_rad >
          scan_conditioning_residual_max_roll_pitch_rad_)
        {
          apply_fallback(
            scan_conditioning_residual_fallback_mode_,
            "residual_motion_limit");
          return finish();
        }
        const std::size_t ratio_minimum = static_cast<std::size_t>(std::ceil(
            scan_conditioning_residual_min_output_valid_ratio_ *
            static_cast<double>(scan->ranges.size())));
        const std::size_t minimum_output_beams = std::max(
          scan_conditioning_residual_min_output_valid_beams_,
          ratio_minimum);
        auto filtered = filterLaserScanByResidual(
          *stable_conditioning_scan_,
          *scan,
          *transform,
          scan_conditioning_residual_threshold_m_,
          scan_conditioning_residual_neighbor_bins_,
          scan_conditioning_residual_min_overlap_beams_,
          scan_conditioning_residual_min_overlap_ratio_,
          minimum_output_beams,
          scan_conditioning_residual_direction_,
          scan_conditioning_residual_max_removed_ratio_);
        if (!filtered) {
          apply_fallback(
            scan_conditioning_residual_fallback_mode_,
            "residual_insufficient_evidence");
          return finish();
        }
        result.output_valid_beams = filtered->output_valid_beam_count;
        result.overlap_beams = filtered->overlap_beam_count;
        result.removed_beams = filtered->removed_beam_count;
        result.scan = std::make_shared<sensor_msgs::msg::LaserScan>(
          std::move(filtered->scan));
        result.action = "residual_filtered_allow";
        ++scan_conditioning_residual_filtered_count_;
        if (scan_conditioning_residual_promote_filtered_reference_) {
          const LaserScanQuality filtered_quality = laserScanQuality(*result.scan);
          rememberStableConditioningScan(
            *result.scan, stamp_ns, filtered_quality);
        }
        return finish();
      }
  }

  result.action = "unknown_mode_fail_open";
  return finish();
}

std::optional<geometry_msgs::msg::TransformStamped>
RoughnessAwareSlamToolbox::stableScanTransform(
  const sensor_msgs::msg::LaserScan & current,
  const sensor_msgs::msg::LaserScan & stable) const
{
  if (
    current.header.frame_id.empty() ||
    stable.header.frame_id.empty() ||
    !isValidMessageTimestamp(
      current.header.stamp.sec, current.header.stamp.nanosec) ||
    !isValidMessageTimestamp(
      stable.header.stamp.sec, stable.header.stamp.nanosec))
  {
    return std::nullopt;
  }
  try {
    const rclcpp::Time current_time(
      current.header.stamp, get_clock()->get_clock_type());
    const rclcpp::Time stable_time(
      stable.header.stamp, get_clock()->get_clock_type());
    return tf_->lookupTransform(
      current.header.frame_id,
      current_time,
      stable.header.frame_id,
      stable_time,
      odom_frame_,
      transform_timeout_);
  } catch (const tf2::TransformException & exception) {
    RCLCPP_DEBUG(
      get_logger(),
      "SLAM-owned exact-time stable scan transform failed: %s",
      exception.what());
    return std::nullopt;
  }
}

void RoughnessAwareSlamToolbox::rememberStableConditioningScan(
  const sensor_msgs::msg::LaserScan & scan,
  const std::int64_t stamp_ns,
  const LaserScanQuality & quality)
{
  const double minimum_ratio =
    scan_conditioning_mode_ == ScanConditioningMode::kStabilize ?
    scan_conditioning_stabilize_min_cached_valid_ratio_ :
    scan_conditioning_residual_min_cached_valid_ratio_;
  if (
    stamp_ns <= 0 ||
    quality.valid_ratio < minimum_ratio ||
    (scan_conditioning_mode_ != ScanConditioningMode::kStabilize &&
    scan_conditioning_mode_ != ScanConditioningMode::kResidualFilter))
  {
    return;
  }
  if (
    stable_conditioning_scan_stamp_ns_ > 0 &&
    stamp_ns < stable_conditioning_scan_stamp_ns_)
  {
    stable_conditioning_scan_.reset();
    stable_conditioning_scan_stamp_ns_ = 0;
  }
  stable_conditioning_scan_ = scan;
  stable_conditioning_scan_stamp_ns_ = stamp_ns;
}

void RoughnessAwareSlamToolbox::resetScanConditioningStateLocked()
{
  stable_conditioning_scan_.reset();
  stable_conditioning_scan_stamp_ns_ = 0;
  scan_conditioning_last_scan_stamp_ns_ = 0;
  scan_conditioning_last_rough_stamp_ns_ = 0;
  scan_conditioning_last_hard_stamp_ns_ = 0;
  scan_conditioning_event_start_stamp_ns_ = 0;
  scan_conditioning_last_publish_stamp_ns_ = 0;
  scan_conditioning_drop_start_stamp_ns_ = 0;
  scan_conditioning_recovery_gate_.reset();
  scan_conditioning_imu_fail_open_gate_.reset();
}

void RoughnessAwareSlamToolbox::resetScanConditioningState(const char * reason)
{
  {
    std::lock_guard<std::mutex> lock(scan_conditioning_mutex_);
    resetScanConditioningStateLocked();
  }
  if (enable_scan_conditioning_) {
    RCLCPP_INFO(
      get_logger(),
      "ROUGHNESS_SCAN_CONDITION_RESET reason=%s",
      reason ? reason : "unknown");
  }
}

const char * RoughnessAwareSlamToolbox::scanConditioningModeName(
  const ScanConditioningMode mode) noexcept
{
  switch (mode) {
    case ScanConditioningMode::kObserve:
      return "observe";
    case ScanConditioningMode::kDrop:
      return "drop";
    case ScanConditioningMode::kThrottle:
      return "throttle";
    case ScanConditioningMode::kQuarantine:
      return "quarantine";
    case ScanConditioningMode::kStabilize:
      return "stabilize";
    case ScanConditioningMode::kResidualFilter:
      return "residual_filter";
  }
  return "unknown";
}

void RoughnessAwareSlamToolbox::resetMapperDerivedState(const char * reason)
{
  std::size_t last_scan_count = 0;
  std::size_t running_sensor_count = 0;
  std::size_t spatial_zone_count = 0;
  std::size_t pose_graph_anchor_count = 0;
  std::size_t roughness_place_event_count = 0;
  {
    boost::mutex::scoped_lock mapper_lock(smapper_mutex_);
    last_scan_count = preview_last_scans_.size();
    running_sensor_count = preview_running_scans_.size();
    preview_last_scans_.clear();
    preview_running_scans_.clear();
  }
  {
    std::lock_guard<std::mutex> state_lock(near_chain_state_mutex_);
    near_chain_last_scan_stamp_ns_ = 0;
    near_chain_recovery_until_stamp_ns_ = 0;
    near_chain_loop_closing_suppression_until_stamp_ns_ = 0;
    adaptive_roughness_memory_.reset();
    roughness_loop_memory_.reset(mapper_generation_.load());
    near_chain_rotation_consistency_.reset();
    near_chain_observer_parity_candidate_streak_ = 0;
    near_chain_observer_parity_trusted_ = false;
    spatial_zone_count = roughness_spatial_zones_.size();
    pose_graph_anchor_count = roughness_pose_graph_anchors_.size();
    roughness_spatial_zones_.clear();
    roughness_pose_graph_anchors_.clear();
  }
  {
    std::lock_guard<std::mutex> place_lock(roughness_place_mutex_);
    roughness_place_event_count = roughness_place_events_.size();
    roughness_place_events_.clear();
    roughness_place_scan_chain_events_.clear();
    roughness_place_injected_passages_.clear();
    roughness_place_reference_chain_observed_passages_.clear();
    roughness_place_event_tracker_.reset();
    roughness_place_scan_chain_tracker_.reset();
  }
  clearPostConfirmation(reason);
  resetScanConditioningState(reason);
  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_MAPPER_STATE_RESET reason=%s mapper_generation=%llu "
    "cleared_last_scans=%zu cleared_running_sensors=%zu "
    "cleared_spatial_zones=%zu cleared_pose_graph_anchors=%zu "
    "cleared_roughness_place_events=%zu",
    reason,
    static_cast<unsigned long long>(mapper_generation_.load()),
    last_scan_count,
    running_sensor_count,
    spatial_zone_count,
    pose_graph_anchor_count,
    roughness_place_event_count);
}

void RoughnessAwareSlamToolbox::logDiagnostics()
{
  const auto now = std::chrono::steady_clock::now();
  if (
    std::chrono::duration<double>(now - last_diagnostics_at_).count() <
    diagnostics_period_s_)
  {
    return;
  }
  std::size_t pending = 0;
  std::size_t queued = 0;
  std::size_t spatial_zones = 0;
  std::size_t roughness_place_events = 0;
  bool roughness_place_event_active = false;
  bool post_confirmation_pending = false;
  {
    std::lock_guard<std::mutex> lock(roughness_mutex_);
    pending = pending_scans_.size();
  }
  {
    std::lock_guard<std::mutex> lock(scan_queue_mutex_);
    queued = scan_queue_.size();
  }
  {
    std::lock_guard<std::mutex> lock(post_confirmation_mutex_);
    post_confirmation_pending = pending_post_confirmation_.has_value();
  }
  {
    std::lock_guard<std::mutex> lock(near_chain_state_mutex_);
    spatial_zones = spatial_memory_use_pose_graph_anchors_ ?
      roughness_pose_graph_anchors_.size() : roughness_spatial_zones_.size();
  }
  {
    std::lock_guard<std::mutex> lock(roughness_place_mutex_);
    roughness_place_events = roughness_place_events_.size();
    roughness_place_event_active = roughness_place_event_tracker_.active();
  }
  const auto received_stamp_audit = received_scan_stamp_audit_.snapshot();
  const auto received_imu_stamp_audit = received_imu_stamp_audit_.snapshot();
  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_INNOVATION_SUMMARY received=%llu queued=%llu complete_windows=%llu "
    "rough=%llu candidates=%llu rejected=%llu unavailable=%llu pending=%zu "
    "decision_cache=%zu scan_queue_depth=%zu scan_worker_in_flight=%llu "
    "scan_worker_completed=%llu duplicate_imu=%llu "
    "out_of_order_imu=%llu "
    "imu_epoch_resets=%llu overflow_fail_open=%llu stale_generation_drops=%llu "
    "exact_reject_configured=%zu "
    "post_armed=%llu post_evaluated=%llu post_confirmed_transient=%llu "
    "post_unavailable=%llu post_expired=%llu post_overlap=%llu post_pending=%u "
    " near_chain_diag=%llu near_chain_rough=%llu "
    "near_chain_spatial_revisit=%llu near_chain_spatial_passage=%llu "
    "near_chain_recovery=%llu "
    "near_chain_incomplete_fail_open=%llu near_chain_timestamp_reset=%llu "
    "near_chain_threshold_override=%llu "
    "near_chain_threshold_restore_failure=%llu "
    "near_chain_loop_closing_suppression=%llu "
    "near_chain_loop_closing_cooldown=%llu "
    "near_chain_loop_closing_restore_failure=%llu near_chain_new_edges=%llu "
    "near_chain_new_nonsequential_edges=%llu "
    "loop_candidates=%llu loop_accepted_callbacks=%llu "
    "loop_actual_edges=%llu loop_accepted_without_edges=%llu "
    "loop_graph_warmups=%llu "
    "loop_pairing_failures=%llu "
    "rotation_batches=%llu rotation_risky_candidates=%llu "
    "rotation_proposed_rejects=%llu rotation_applied_rejects=%llu "
    "rotation_fail_open=%llu rotation_timestamp_resets=%llu "
    "spatial_zones=%zu "
    "spatial_zone_inserts=%llu spatial_zone_merges=%llu "
    "spatial_zone_evictions=%llu spatial_invalid_pose=%llu "
    "spatial_pose_graph_refreshes=%llu spatial_pose_graph_stale_removes=%llu "
    "spatial_pose_graph_invalid_removes=%llu "
    "spatial_pose_graph_prediction_failures=%llu "
    "spatial_pose_graph_anchor_inserts=%llu "
    "scan_conditioning_allow=%llu scan_conditioning_drop=%llu "
    "scan_conditioning_bridge=%llu scan_conditioning_quarantine=%llu "
    "scan_conditioning_stabilized=%llu "
    "scan_conditioning_residual_filtered=%llu "
    "scan_conditioning_fallback=%llu "
    "scan_conditioning_imu_quality_reject=%llu "
    "scan_conditioning_tf_failure=%llu "
    "received_first_stamp_ns=%lld received_last_stamp_ns=%lld "
    "received_max_positive_gap_ns=%lld "
    "received_stamp_duplicate_count=%llu "
    "received_stamp_out_of_order_count=%llu "
    "received_stamp_fingerprint_fnv1a64=%016llx "
    "received_imu=%llu received_imu_first_stamp_ns=%lld "
    "received_imu_last_stamp_ns=%lld received_imu_max_positive_gap_ns=%lld "
    "received_imu_stamp_duplicate_count=%llu "
    "received_imu_stamp_out_of_order_count=%llu "
    "received_imu_stamp_fingerprint_fnv1a64=%016llx",
    static_cast<unsigned long long>(received_scans_.load()),
    static_cast<unsigned long long>(queued_scans_.load()),
    static_cast<unsigned long long>(complete_windows_.load()),
    static_cast<unsigned long long>(rough_scans_.load()),
    static_cast<unsigned long long>(innovation_candidates_.load()),
    static_cast<unsigned long long>(rejected_scans_.load()),
    static_cast<unsigned long long>(unavailable_innovations_.load()),
    pending,
    std::size_t{0},
    queued,
    static_cast<unsigned long long>(scan_worker_in_flight_.load()),
    static_cast<unsigned long long>(scan_worker_completed_.load()),
    static_cast<unsigned long long>(duplicate_imu_samples_.load()),
    static_cast<unsigned long long>(out_of_order_imu_samples_.load()),
    static_cast<unsigned long long>(imu_epoch_resets_.load()),
    static_cast<unsigned long long>(pending_overflow_fail_open_.load()),
    static_cast<unsigned long long>(
      stale_mapper_generation_scan_drop_count_.load()),
    experimental_reject_scan_stamps_ns_.size(),
    static_cast<unsigned long long>(post_confirmation_armed_.load()),
    static_cast<unsigned long long>(post_confirmation_evaluated_.load()),
    static_cast<unsigned long long>(
      post_confirmation_confirmed_transient_.load()),
    static_cast<unsigned long long>(post_confirmation_unavailable_.load()),
    static_cast<unsigned long long>(post_confirmation_expired_.load()),
    static_cast<unsigned long long>(post_confirmation_overlap_.load()),
    post_confirmation_pending ? 1U : 0U,
    static_cast<unsigned long long>(near_chain_diag_count_.load()),
    static_cast<unsigned long long>(near_chain_rough_gate_count_.load()),
    static_cast<unsigned long long>(
      near_chain_spatial_revisit_gate_count_.load()),
    static_cast<unsigned long long>(
      near_chain_spatial_passage_gate_count_.load()),
    static_cast<unsigned long long>(near_chain_recovery_gate_count_.load()),
    static_cast<unsigned long long>(
      near_chain_incomplete_fail_open_count_.load()),
    static_cast<unsigned long long>(near_chain_timestamp_reset_count_.load()),
    static_cast<unsigned long long>(
      near_chain_threshold_override_count_.load()),
    static_cast<unsigned long long>(
      near_chain_threshold_restore_failure_count_.load()),
    static_cast<unsigned long long>(
      near_chain_loop_closing_suppression_count_.load()),
    static_cast<unsigned long long>(
      near_chain_loop_closing_cooldown_count_.load()),
    static_cast<unsigned long long>(
      near_chain_loop_closing_restore_failure_count_.load()),
    static_cast<unsigned long long>(near_chain_new_edge_count_.load()),
    static_cast<unsigned long long>(
      near_chain_new_nonsequential_edge_count_.load()),
    static_cast<unsigned long long>(loop_closure_candidate_count_.load()),
    static_cast<unsigned long long>(loop_closure_accepted_count_.load()),
    static_cast<unsigned long long>(loop_closure_actual_edge_count_.load()),
    static_cast<unsigned long long>(
      loop_closure_accepted_without_edge_count_.load()),
    static_cast<unsigned long long>(loop_closure_graph_warmup_count_.load()),
    static_cast<unsigned long long>(
      loop_closure_pairing_failure_count_.load()),
    static_cast<unsigned long long>(near_chain_rotation_batch_count_.load()),
    static_cast<unsigned long long>(
      near_chain_rotation_risky_candidate_count_.load()),
    static_cast<unsigned long long>(
      near_chain_rotation_proposed_reject_count_.load()),
    static_cast<unsigned long long>(
      near_chain_rotation_applied_reject_count_.load()),
    static_cast<unsigned long long>(
      near_chain_rotation_fail_open_count_.load()),
    static_cast<unsigned long long>(
      near_chain_rotation_timestamp_reset_count_.load()),
    spatial_zones,
    static_cast<unsigned long long>(spatial_memory_zone_insert_count_.load()),
    static_cast<unsigned long long>(spatial_memory_zone_merge_count_.load()),
    static_cast<unsigned long long>(spatial_memory_zone_evict_count_.load()),
    static_cast<unsigned long long>(spatial_memory_invalid_pose_count_.load()),
    static_cast<unsigned long long>(
      spatial_memory_pose_graph_refresh_count_.load()),
    static_cast<unsigned long long>(
      spatial_memory_pose_graph_stale_remove_count_.load()),
    static_cast<unsigned long long>(
      spatial_memory_pose_graph_invalid_remove_count_.load()),
    static_cast<unsigned long long>(
      spatial_memory_pose_graph_prediction_failure_count_.load()),
    static_cast<unsigned long long>(
      spatial_memory_pose_graph_anchor_insert_count_.load()),
    static_cast<unsigned long long>(scan_conditioning_allow_count_.load()),
    static_cast<unsigned long long>(scan_conditioning_drop_count_.load()),
    static_cast<unsigned long long>(scan_conditioning_bridge_count_.load()),
    static_cast<unsigned long long>(scan_conditioning_quarantine_count_.load()),
    static_cast<unsigned long long>(scan_conditioning_stabilized_count_.load()),
    static_cast<unsigned long long>(
      scan_conditioning_residual_filtered_count_.load()),
    static_cast<unsigned long long>(scan_conditioning_fallback_count_.load()),
    static_cast<unsigned long long>(
      scan_conditioning_imu_quality_reject_count_.load()),
    static_cast<unsigned long long>(scan_conditioning_tf_failure_count_.load()),
    static_cast<long long>(received_stamp_audit.first_stamp_ns),
    static_cast<long long>(received_stamp_audit.last_stamp_ns),
    static_cast<long long>(received_stamp_audit.max_positive_gap_ns),
    static_cast<unsigned long long>(received_stamp_audit.duplicate_count),
    static_cast<unsigned long long>(received_stamp_audit.out_of_order_count),
    static_cast<unsigned long long>(
      received_stamp_audit.fingerprint_fnv1a64),
    static_cast<unsigned long long>(received_imus_.load()),
    static_cast<long long>(received_imu_stamp_audit.first_stamp_ns),
    static_cast<long long>(received_imu_stamp_audit.last_stamp_ns),
    static_cast<long long>(received_imu_stamp_audit.max_positive_gap_ns),
    static_cast<unsigned long long>(received_imu_stamp_audit.duplicate_count),
    static_cast<unsigned long long>(received_imu_stamp_audit.out_of_order_count),
    static_cast<unsigned long long>(
      received_imu_stamp_audit.fingerprint_fnv1a64));
  RCLCPP_INFO(
    get_logger(),
    "ROUGHNESS_PLACE_SUMMARY enabled=%s events_started=%llu events_finalized=%llu "
    "valid_descriptors=%llu candidates_available=%llu timestamp_resets=%llu "
    "scan_match_observer_enabled=%s scan_match_attempted=%llu "
    "scan_match_available=%llu scan_match_fail_open=%llu "
    "direction_hypotheses=%llu direction_overrides=%llu "
    "direction_fail_open=%llu "
    "reference_chain_enabled=%s reference_chain_attempted=%llu "
    "reference_chain_available=%llu reference_chain_fail_open=%llu "
    "reference_chain_event_accepted=%llu "
    "reference_chain_correlated_duplicates=%llu "
    "quality_accepted=%llu quality_rejected=%llu quality_fail_open=%llu "
    "constraint_mode=%s constraints_injected=%llu constraint_duplicates=%llu "
    "constraint_correlated_duplicates=%llu constraint_failures=%llu "
    "memory_events=%zu active_event=%s",
    enable_roughness_place_recognition_observer_ ? "true" : "false",
    static_cast<unsigned long long>(roughness_place_event_started_.load()),
    static_cast<unsigned long long>(roughness_place_event_finalized_.load()),
    static_cast<unsigned long long>(roughness_place_descriptor_valid_.load()),
    static_cast<unsigned long long>(
      roughness_place_candidate_available_.load()),
    static_cast<unsigned long long>(roughness_place_timestamp_reset_.load()),
    enable_roughness_place_scan_match_observer_ ? "true" : "false",
    static_cast<unsigned long long>(
      roughness_place_scan_match_attempted_.load()),
    static_cast<unsigned long long>(
      roughness_place_scan_match_available_.load()),
    static_cast<unsigned long long>(
      roughness_place_scan_match_fail_open_.load()),
    static_cast<unsigned long long>(
      roughness_place_direction_hypotheses_.load()),
    static_cast<unsigned long long>(
      roughness_place_direction_overrides_.load()),
    static_cast<unsigned long long>(
      roughness_place_direction_fail_open_.load()),
    roughness_place_enable_reference_chain_observer_ ? "true" : "false",
    static_cast<unsigned long long>(
      roughness_place_reference_chain_attempted_.load()),
    static_cast<unsigned long long>(
      roughness_place_reference_chain_available_.load()),
    static_cast<unsigned long long>(
      roughness_place_reference_chain_fail_open_.load()),
    static_cast<unsigned long long>(
      roughness_place_reference_chain_event_accepted_.load()),
    static_cast<unsigned long long>(
      roughness_place_reference_chain_correlated_duplicate_.load()),
    static_cast<unsigned long long>(roughness_place_quality_accepted_.load()),
    static_cast<unsigned long long>(roughness_place_quality_rejected_.load()),
    static_cast<unsigned long long>(roughness_place_quality_fail_open_.load()),
    roughness_place_constraint_mode_.c_str(),
    static_cast<unsigned long long>(
      roughness_place_constraint_injected_.load()),
    static_cast<unsigned long long>(
      roughness_place_constraint_duplicate_.load()),
    static_cast<unsigned long long>(
      roughness_place_constraint_correlated_duplicate_.load()),
    static_cast<unsigned long long>(
      roughness_place_constraint_failure_.load()),
    roughness_place_events,
    roughness_place_event_active ? "true" : "false");
  last_diagnostics_at_ = now;
}

std::int64_t RoughnessAwareSlamToolbox::messageStampNs(
  const builtin_interfaces::msg::Time & stamp)
{
  return
    static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
    static_cast<std::int64_t>(stamp.nanosec);
}

double RoughnessAwareSlamToolbox::tiltDeg(const sensor_msgs::msg::Imu & msg)
{
  const auto & q = msg.orientation;
  if (
    !std::isfinite(q.x) || !std::isfinite(q.y) ||
    !std::isfinite(q.z) || !std::isfinite(q.w))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double norm =
    std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm <= 1.0e-12) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double x = q.x / norm;
  const double y = q.y / norm;
  const double z = q.z / norm;
  const double w = q.w / norm;
  const double roll = std::atan2(
    2.0 * (w * x + y * z),
    1.0 - 2.0 * (x * x + y * y));
  const double pitch = std::asin(
    std::clamp(
      2.0 * (w * y - z * x), -1.0, 1.0));
  return std::hypot(roll, pitch) * kRadiansToDegrees;
}

double RoughnessAwareSlamToolbox::normalizedAngle(const double angle_rad)
{
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

}  // namespace slam_2d_core
