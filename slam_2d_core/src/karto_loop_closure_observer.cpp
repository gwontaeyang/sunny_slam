#include "slam_2d_core/karto_loop_closure_observer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

namespace slam_2d_core
{
namespace
{

double normalizedAngle(const double angle_rad) noexcept
{
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

bool finitePose(const karto::Pose2 & pose) noexcept
{
  return std::isfinite(pose.GetX()) && std::isfinite(pose.GetY()) &&
         std::isfinite(pose.GetHeading());
}

void setPoseUnavailable(karto::Pose2 & pose) noexcept
{
  const double unavailable = std::numeric_limits<double>::quiet_NaN();
  pose.SetX(unavailable);
  pose.SetY(unavailable);
  pose.SetHeading(unavailable);
}

void setMatrixUnavailable(karto::Matrix3 & matrix) noexcept
{
  const double unavailable = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t row = 0; row < 3U; ++row) {
    for (std::size_t column = 0; column < 3U; ++column) {
      matrix(row, column) = unavailable;
    }
  }
}

void initializeUnavailableMeasurements(KartoLoopClosureAttempt & attempt) noexcept
{
  setPoseUnavailable(attempt.relative_transform);
  setPoseUnavailable(attempt.odometric_relative_transform);
  setPoseUnavailable(attempt.pre_graph_relative_transform);
  setPoseUnavailable(attempt.post_graph_relative_transform);
  setMatrixUnavailable(attempt.factor_covariance);
}

karto::Pose2 relativePose(
  const karto::Pose2 & source,
  const karto::Pose2 & target)
{
  karto::Transform transform(source, karto::Pose2());
  return transform.TransformPose(target);
}

void setResidual(
  const karto::Pose2 & expected,
  const karto::Pose2 & measured,
  double & x,
  double & y,
  double & yaw) noexcept
{
  x = expected.GetX() - measured.GetX();
  y = expected.GetY() - measured.GetY();
  yaw = normalizedAngle(expected.GetHeading() - measured.GetHeading());
}

double covarianceWeightedSquared(
  const double x,
  const double y,
  const double yaw,
  const karto::Matrix3 & covariance) noexcept
{
  karto::Matrix3 regularized = covariance;
  regularized(0, 0) += 1.0e-9;
  regularized(1, 1) += 1.0e-9;
  regularized(2, 2) += 1.0e-9;
  karto::Matrix3 information;
  if (!regularized.InverseFast(information, 1.0e-14)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double residual[3] = {x, y, yaw};
  double value = 0.0;
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      value += residual[row] * information(row, column) * residual[column];
    }
  }
  return std::isfinite(value) ? value :
         std::numeric_limits<double>::quiet_NaN();
}

bool contains(
  const std::vector<std::int32_t> & values,
  const std::int32_t value) noexcept
{
  return std::find(values.begin(), values.end(), value) != values.end();
}

const KartoLoopClosureSourceSnapshot * findSnapshot(
  const KartoLoopClosureAttempt & attempt,
  const std::int32_t unique_id) noexcept
{
  const auto found = std::find_if(
    attempt.source_snapshots.begin(), attempt.source_snapshots.end(),
    [unique_id](const KartoLoopClosureSourceSnapshot & snapshot) {
      return snapshot.unique_id == unique_id;
    });
  return found == attempt.source_snapshots.end() ? nullptr : &*found;
}

karto::LocalizedRangeScanVector findPossibleLoopClosureCandidate(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * current_scan,
  const karto::Name & sensor_name,
  kt_int32u & scan_index)
{
  karto::LocalizedRangeScanVector chain;
  auto * graph = mapper->GetGraph();
  auto * sensor_manager = mapper->GetMapperSensorManager();
  const bool use_scan_barycenter = mapper->getParamUseScanBarycenter();
  const double maximum_distance = mapper->getParamLoopSearchMaximumDistance();
  const double maximum_distance_squared = maximum_distance * maximum_distance;
  const karto::Pose2 pose = current_scan->GetReferencePose(use_scan_barycenter);
  const auto near_linked_scans = graph->FindNearLinkedScans(
    current_scan, maximum_distance);
  // Mapper::Process assigns monotonically increasing per-sensor state IDs
  // before TryCloseLoop. In the supported single-sensor mapping path this is
  // the same upper bound used by Karto's private GetScans(...).size(). The
  // latter is declared in the SDK header but not exported by the shared
  // library, so calling it here would leave downstream test binaries with an
  // unresolved symbol.
  if (current_scan->GetStateId() < 0) {
    return chain;
  }
  const auto scan_count = static_cast<kt_int32u>(
    current_scan->GetStateId() + 1);
  const auto minimum_chain_size = static_cast<std::size_t>(
    std::max(mapper->getParamLoopMatchMinimumChainSize(), 0));

  for (; scan_index < scan_count; ++scan_index) {
    auto * candidate_scan = sensor_manager->GetScan(sensor_name, scan_index);
    if (!candidate_scan) {
      continue;
    }
    const karto::Pose2 candidate_pose = candidate_scan->GetReferencePose(
      use_scan_barycenter);
    const double squared_distance = candidate_pose.GetPosition().SquaredDistance(
      pose.GetPosition());
    if (squared_distance < maximum_distance_squared + karto::KT_TOLERANCE) {
      if (std::find(
          near_linked_scans.begin(), near_linked_scans.end(),
          candidate_scan) != near_linked_scans.end())
      {
        chain.clear();
      } else {
        chain.push_back(candidate_scan);
      }
    } else if (chain.size() >= minimum_chain_size) {
      return chain;
    } else {
      chain.clear();
    }
  }
  return chain;
}

}  // namespace

const char * kartoLoopClosureDispositionName(
  const KartoLoopClosureDisposition disposition) noexcept
{
  switch (disposition) {
    case KartoLoopClosureDisposition::kPending:
      return "pending";
    case KartoLoopClosureDisposition::kCoarseRejected:
      return "coarse_rejected";
    case KartoLoopClosureDisposition::kFineRejected:
      return "fine_rejected";
    case KartoLoopClosureDisposition::kAcceptedIncomplete:
      return "accepted_incomplete";
    case KartoLoopClosureDisposition::kAccepted:
      return "accepted";
  }
  return "unknown";
}

bool parseKartoLoopClosureCoarseMessage(
  const std::string & message,
  double & response,
  double & response_threshold,
  double & covariance_xx,
  double & covariance_yy,
  double & covariance_threshold) noexcept
{
  const int parsed = std::sscanf(
    message.c_str(),
    "COARSE RESPONSE: %lf (> %lf) var: %lf, %lf (< %lf)",
    &response,
    &response_threshold,
    &covariance_xx,
    &covariance_yy,
    &covariance_threshold);
  return parsed == 5 && std::isfinite(response) &&
         std::isfinite(response_threshold) && std::isfinite(covariance_xx) &&
         std::isfinite(covariance_yy) && std::isfinite(covariance_threshold);
}

bool parseKartoLoopClosureFineMessage(
  const std::string & message,
  double & response,
  double & response_threshold) noexcept
{
  const int parsed = std::sscanf(
    message.c_str(),
    "FINE RESPONSE: %lf (>%lf)",
    &response,
    &response_threshold);
  return parsed == 2 && std::isfinite(response) &&
         std::isfinite(response_threshold);
}

KartoLoopClosureObserver::KartoLoopClosureObserver(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * current_scan,
  const std::int64_t stamp_ns)
{
  (void)attach(mapper, current_scan, stamp_ns);
}

KartoLoopClosureObserver::~KartoLoopClosureObserver()
{
  (void)detach();
}

bool KartoLoopClosureObserver::attach(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * current_scan,
  const std::int64_t stamp_ns) noexcept
{
  if (attached_ || finished_) {
    return false;
  }
  batch_.observer_requested = true;
  batch_.stamp_ns = stamp_ns;
  mapper_ = mapper;
  current_scan_ = current_scan;
  if (!mapper_ || !current_scan_ || !mapper_->GetGraph()) {
    appendFailure("graph_not_initialized");
    return false;
  }
  try {
    const auto sensor_names =
      mapper_->GetMapperSensorManager()->GetSensorNames();
    batch_.single_sensor_context =
      sensor_names.size() <= 1U &&
      (sensor_names.empty() || sensor_names.front() == current_scan_->GetSensorName());
    if (!batch_.single_sensor_context) {
      appendFailure("multi_sensor_candidate_identity_unsupported");
      return false;
    }
    mapper_->AddListener(this);
    attached_ = true;
    batch_.listener_attached = true;
    batch_.available = true;
    return true;
  } catch (...) {
    appendFailure("listener_attach_exception");
    return false;
  }
}

bool KartoLoopClosureObserver::detach() noexcept
{
  if (!attached_) {
    return batch_.listener_detached || !batch_.listener_attached;
  }
  try {
    mapper_->RemoveListener(this);
    attached_ = false;
    batch_.listener_detached = true;
    return true;
  } catch (...) {
    attached_ = false;
    appendFailure("listener_detach_exception");
    return false;
  }
}

KartoLoopClosureObservationBatch KartoLoopClosureObserver::finish() noexcept
{
  if (!finished_) {
    (void)detach();
    finalizeOpenAttempt();
    batch_.candidate_identity_complete = std::all_of(
      batch_.attempts.begin(), batch_.attempts.end(),
      [](const KartoLoopClosureAttempt & attempt) {
        return attempt.candidate_chain_available;
      });
    batch_.listener_sequence_complete = std::all_of(
      batch_.attempts.begin(), batch_.attempts.end(),
      [](const KartoLoopClosureAttempt & attempt) {
        return attempt.listener_sequence_complete;
      });
    batch_.accepted_callback_count = static_cast<std::size_t>(std::count_if(
        batch_.attempts.begin(), batch_.attempts.end(),
        [](const KartoLoopClosureAttempt & attempt) {
          return attempt.disposition == KartoLoopClosureDisposition::kAccepted;
        }));
    finished_ = true;
  }
  return batch_;
}

void KartoLoopClosureObserver::LoopClosureCheck(const std::string & info)
{
  try {
    if (info.rfind("COARSE RESPONSE:", 0) == 0) {
      finalizeOpenAttempt();
      KartoLoopClosureAttempt attempt;
      initializeUnavailableMeasurements(attempt);
      attempt.candidate_index = batch_.attempts.size();
      attempt.sensor_name = current_scan_ ?
        current_scan_->GetSensorName().ToString() : std::string();
      attempt.coarse_message_parsed = parseKartoLoopClosureCoarseMessage(
        info,
        attempt.coarse_response,
        attempt.coarse_response_threshold,
        attempt.coarse_covariance_xx_m2,
        attempt.coarse_covariance_yy_m2,
        attempt.coarse_covariance_threshold_m2);
      if (!attempt.coarse_message_parsed) {
        appendFailure("coarse_message_parse_failure");
      } else {
        attempt.coarse_accepted =
          attempt.coarse_response > attempt.coarse_response_threshold &&
          attempt.coarse_covariance_xx_m2 <
          attempt.coarse_covariance_threshold_m2 &&
          attempt.coarse_covariance_yy_m2 <
          attempt.coarse_covariance_threshold_m2;
      }
      captureCandidateChain(attempt);
      if (attempt.coarse_message_parsed && !attempt.coarse_accepted) {
        attempt.disposition = KartoLoopClosureDisposition::kCoarseRejected;
        attempt.listener_sequence_complete = true;
      }
      batch_.attempts.push_back(std::move(attempt));
      return;
    }

    if (info.rfind("FINE RESPONSE:", 0) == 0) {
      if (batch_.attempts.empty()) {
        appendFailure("orphan_fine_callback");
        return;
      }
      auto & attempt = batch_.attempts.back();
      attempt.fine_message_parsed = parseKartoLoopClosureFineMessage(
        info, attempt.fine_response, attempt.fine_response_threshold);
      if (!attempt.fine_message_parsed) {
        appendFailure("fine_message_parse_failure");
      } else {
        attempt.fine_accepted =
          attempt.fine_response >= attempt.fine_response_threshold;
      }
      return;
    }

    if (info.rfind("REJECTED!", 0) == 0) {
      if (batch_.attempts.empty()) {
        appendFailure("orphan_rejection_callback");
        return;
      }
      auto & attempt = batch_.attempts.back();
      attempt.rejection_callback_seen = true;
      attempt.disposition = KartoLoopClosureDisposition::kFineRejected;
      attempt.listener_sequence_complete =
        attempt.coarse_message_parsed && attempt.coarse_accepted &&
        attempt.fine_message_parsed && !attempt.fine_accepted;
      if (!attempt.listener_sequence_complete) {
        appendFailure("inconsistent_rejection_sequence");
      }
    }
  } catch (...) {
    appendFailure("loop_check_callback_exception");
  }
}

void KartoLoopClosureObserver::BeginLoopClosure(const std::string &)
{
  try {
    if (batch_.attempts.empty()) {
      appendFailure("orphan_begin_callback");
      return;
    }
    auto & attempt = batch_.attempts.back();
    attempt.begin_callback_seen = true;
    attempt.disposition = KartoLoopClosureDisposition::kAcceptedIncomplete;
    if (
      !attempt.coarse_message_parsed || !attempt.coarse_accepted ||
      !attempt.fine_message_parsed || !attempt.fine_accepted)
    {
      appendFailure("inconsistent_begin_sequence");
    }
    if (mapper_ && mapper_->GetGraph()) {
      attempt.event_edge_count_before = mapper_->GetGraph()->GetEdges().size();
      attempt.event_edge_count_before_available = true;
    } else if (attached_) {
      appendFailure("loop_event_graph_unavailable_at_begin");
    }
  } catch (...) {
    appendFailure("loop_begin_callback_exception");
  }
}

void KartoLoopClosureObserver::EndLoopClosure(const std::string &)
{
  try {
    if (batch_.attempts.empty()) {
      appendFailure("orphan_end_callback");
      return;
    }
    auto & attempt = batch_.attempts.back();
    attempt.end_callback_seen = true;
    captureAcceptedOutcome(attempt);
    attempt.disposition = KartoLoopClosureDisposition::kAccepted;
    attempt.listener_sequence_complete =
      attempt.coarse_message_parsed && attempt.coarse_accepted &&
      attempt.fine_message_parsed && attempt.fine_accepted &&
      attempt.begin_callback_seen;
    if (!attempt.listener_sequence_complete) {
      appendFailure("inconsistent_end_sequence");
    }
  } catch (...) {
    appendFailure("loop_end_callback_exception");
  }
}

void KartoLoopClosureObserver::appendFailure(const std::string & reason) noexcept
{
  if (reason.empty()) {
    return;
  }
  if (batch_.failure_reason.empty()) {
    batch_.failure_reason = reason;
  } else if (batch_.failure_reason.find(reason) == std::string::npos) {
    batch_.failure_reason += ";" + reason;
  }
}

void KartoLoopClosureObserver::finalizeOpenAttempt() noexcept
{
  if (batch_.attempts.empty()) {
    return;
  }
  auto & attempt = batch_.attempts.back();
  if (attempt.listener_sequence_complete) {
    return;
  }
  if (attempt.coarse_message_parsed && !attempt.coarse_accepted) {
    attempt.disposition = KartoLoopClosureDisposition::kCoarseRejected;
    attempt.listener_sequence_complete = true;
    return;
  }
  if (
    attempt.fine_message_parsed && !attempt.fine_accepted &&
    attempt.rejection_callback_seen)
  {
    attempt.disposition = KartoLoopClosureDisposition::kFineRejected;
    attempt.listener_sequence_complete = true;
    return;
  }
  if (attempt.begin_callback_seen && !attempt.end_callback_seen) {
    attempt.disposition = KartoLoopClosureDisposition::kAcceptedIncomplete;
  }
  appendFailure("incomplete_listener_sequence");
}

void KartoLoopClosureObserver::captureCandidateChain(
  KartoLoopClosureAttempt & attempt) noexcept
{
  if (
    !batch_.single_sensor_context || !mapper_ || !current_scan_ ||
    !mapper_->GetGraph())
  {
    appendFailure("candidate_chain_context_unavailable");
    return;
  }
  try {
    const auto chain = findPossibleLoopClosureCandidate(
      mapper_, current_scan_, current_scan_->GetSensorName(),
      candidate_scan_index_);
    if (chain.empty()) {
      appendFailure("candidate_chain_replay_empty");
      return;
    }
    attempt.pre_target_corrected_pose = current_scan_->GetCorrectedPose();
    attempt.pre_target_odometric_pose = current_scan_->GetOdometricPose();
    attempt.source_snapshots.reserve(chain.size());
    for (auto * scan : chain) {
      if (!scan) {
        appendFailure("candidate_chain_contains_null_scan");
        return;
      }
      attempt.chain_state_ids.push_back(scan->GetStateId());
      attempt.chain_unique_ids.push_back(scan->GetUniqueId());
      attempt.source_snapshots.push_back(
        KartoLoopClosureSourceSnapshot{
          scan->GetStateId(),
          scan->GetUniqueId(),
          scan->GetCorrectedPose(),
          scan->GetOdometricPose()});
    }
    attempt.candidate_chain_available = true;
  } catch (...) {
    appendFailure("candidate_chain_replay_exception");
  }
}

void KartoLoopClosureObserver::captureAcceptedOutcome(
  KartoLoopClosureAttempt & attempt) noexcept
{
  if (!mapper_ || !current_scan_ || !mapper_->GetGraph()) {
    if (attached_) {
      appendFailure("loop_event_graph_unavailable_at_end");
    }
    return;
  }
  if (!attempt.event_edge_count_before_available) {
    if (attached_) {
      appendFailure("loop_event_edge_count_before_unavailable");
    }
    return;
  }

  try {
    const auto & edges = mapper_->GetGraph()->GetEdges();
    attempt.event_edge_count_after = edges.size();
    if (attempt.event_edge_count_after < attempt.event_edge_count_before) {
      appendFailure("loop_event_edge_count_rollback");
      return;
    }
    attempt.event_new_edge_count =
      attempt.event_edge_count_after - attempt.event_edge_count_before;
    if (attempt.event_new_edge_count == 0U) {
      // Karto may accept the scan match but skip LinkScans when the closest
      // chain scan is outside LinkScanMaximumDistance.
      attempt.accepted_without_edge = true;
      return;
    }

    std::vector<karto::Edge<karto::LocalizedRangeScan> *> matching_edges;
    for (std::size_t edge_index = attempt.event_edge_count_before;
      edge_index < attempt.event_edge_count_after; ++edge_index)
    {
      auto * edge = edges[edge_index];
      if (
        !edge || !edge->GetSource() || !edge->GetTarget() ||
        !edge->GetSource()->GetObject() || !edge->GetTarget()->GetObject())
      {
        appendFailure("loop_event_malformed_new_edge");
        continue;
      }
      auto * edge_source = edge->GetSource()->GetObject();
      auto * edge_target = edge->GetTarget()->GetObject();
      if (
        edge_target->GetUniqueId() == current_scan_->GetUniqueId() &&
        contains(attempt.chain_unique_ids, edge_source->GetUniqueId()))
      {
        matching_edges.push_back(edge);
      }
    }

    if (matching_edges.size() != 1U || attempt.event_new_edge_count != 1U) {
      attempt.edge_pairing_ambiguous = matching_edges.size() > 1U;
      appendFailure(
        matching_edges.empty() ? "loop_event_edge_not_identified" :
        "loop_event_edge_count_ambiguous");
      return;
    }

    auto * edge = matching_edges.front();
    attempt.event_edge_index = attempt.event_edge_count_before;
    auto * edge_source = edge->GetSource()->GetObject();
    auto * edge_target = edge->GetTarget()->GetObject();
    attempt.source_state_id = edge_source->GetStateId();
    attempt.source_unique_id = edge_source->GetUniqueId();
    attempt.target_state_id = edge_target->GetStateId();
    attempt.target_unique_id = edge_target->GetUniqueId();
    attempt.actual_edge_present = true;
    auto * link = dynamic_cast<karto::LinkInfo *>(edge->GetLabel());
    attempt.link_info_available = link != nullptr;
    if (!link) {
      appendFailure("loop_event_link_info_unavailable");
      return;
    }

    attempt.relative_transform = link->GetPoseDifference();
    attempt.factor_covariance = link->GetCovariance();
    const auto * source_snapshot = findSnapshot(
      attempt, edge_source->GetUniqueId());
    attempt.odometric_relative_transform = relativePose(
      source_snapshot ? source_snapshot->odometric_pose :
      edge_source->GetOdometricPose(),
      attempt.pre_target_odometric_pose);
    attempt.post_graph_relative_transform = relativePose(
      edge_source->GetCorrectedPose(), edge_target->GetCorrectedPose());
    if (source_snapshot && finitePose(source_snapshot->corrected_pose)) {
      attempt.pre_graph_relative_transform = relativePose(
        source_snapshot->corrected_pose, attempt.pre_target_corrected_pose);
    }

    const double odom_dx =
      attempt.odometric_relative_transform.GetX() -
      attempt.relative_transform.GetX();
    const double odom_dy =
      attempt.odometric_relative_transform.GetY() -
      attempt.relative_transform.GetY();
    const double odom_dyaw = normalizedAngle(
      attempt.odometric_relative_transform.GetHeading() -
      attempt.relative_transform.GetHeading());
    attempt.measurement_vs_odometry_translation_m = std::hypot(odom_dx, odom_dy);
    attempt.measurement_vs_odometry_yaw_rad = std::fabs(odom_dyaw);
    attempt.measurement_vs_odometry_covariance_weighted_disagreement_sq =
      covarianceWeightedSquared(
      odom_dx, odom_dy, odom_dyaw, attempt.factor_covariance);
    if (source_snapshot) {
      setResidual(
        attempt.pre_graph_relative_transform,
        attempt.relative_transform,
        attempt.pre_residual_x_m,
        attempt.pre_residual_y_m,
        attempt.pre_residual_yaw_rad);
    }
    setResidual(
      attempt.post_graph_relative_transform,
      attempt.relative_transform,
      attempt.post_residual_x_m,
      attempt.post_residual_y_m,
      attempt.post_residual_yaw_rad);
    attempt.post_covariance_weighted_residual_sq = covarianceWeightedSquared(
      attempt.post_residual_x_m,
      attempt.post_residual_y_m,
      attempt.post_residual_yaw_rad,
      attempt.factor_covariance);
  } catch (...) {
    appendFailure("loop_event_outcome_exception");
  }
}

void updateKartoLoopClosureOutcomes(
  KartoLoopClosureObservationBatch & batch,
  karto::Mapper * mapper,
  const karto::LocalizedRangeScan * processed_scan) noexcept
{
  batch.edge_pairing_complete = false;
  batch.actual_loop_edge_count = 0;
  batch.accepted_without_edge_count = 0;
  batch.ambiguous_loop_edge_count = 0;
  batch.unpaired_loop_edge_count = 0;
  if (!batch.available || !mapper || !processed_scan || !mapper->GetGraph()) {
    return;
  }
  batch.target_state_id = processed_scan->GetStateId();
  batch.target_unique_id = processed_scan->GetUniqueId();
  bool every_accepted_accounted_for = true;
  for (const auto & attempt : batch.attempts) {
    if (attempt.disposition != KartoLoopClosureDisposition::kAccepted) {
      continue;
    }
    if (attempt.edge_pairing_ambiguous) {
      ++batch.ambiguous_loop_edge_count;
      every_accepted_accounted_for = false;
    } else if (attempt.actual_edge_present && attempt.link_info_available) {
      ++batch.actual_loop_edge_count;
    } else if (attempt.accepted_without_edge) {
      ++batch.accepted_without_edge_count;
    } else {
      ++batch.unpaired_loop_edge_count;
      every_accepted_accounted_for = false;
    }
  }
  batch.edge_pairing_complete =
    batch.candidate_identity_complete && batch.listener_sequence_complete &&
    batch.ambiguous_loop_edge_count == 0U &&
    batch.unpaired_loop_edge_count == 0U && every_accepted_accounted_for &&
    batch.actual_loop_edge_count + batch.accepted_without_edge_count ==
    batch.accepted_callback_count;
  if (!batch.edge_pairing_complete) {
    if (batch.failure_reason.empty()) {
      batch.failure_reason = "loop_edge_pairing_incomplete";
    } else if (
      batch.failure_reason.find("loop_edge_pairing_incomplete") ==
      std::string::npos)
    {
      batch.failure_reason += ";loop_edge_pairing_incomplete";
    }
  }
}

}  // namespace slam_2d_core
