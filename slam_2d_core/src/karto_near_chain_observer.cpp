#include "slam_2d_core/karto_near_chain_observer.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <list>
#include <set>
#include <unordered_set>
#include <utility>

namespace slam_2d_core
{
namespace
{

constexpr double kKartoTolerance = 1.0e-6;

double normalizedAngle(const double angle_rad) noexcept
{
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

karto::Vertex<karto::LocalizedRangeScan> * findVertex(
  karto::MapperGraph * graph,
  const karto::Name & sensor_name,
  const std::int32_t state_id) noexcept
{
  if (!graph) {
    return nullptr;
  }
  const auto & vertices = graph->GetVertices();
  const auto sensor = vertices.find(sensor_name);
  if (sensor == vertices.end()) {
    return nullptr;
  }
  const auto vertex = sensor->second.find(state_id);
  return vertex == sensor->second.end() ? nullptr : vertex->second;
}

karto::LocalizedRangeScan * findScan(
  karto::MapperGraph * graph,
  const karto::Name & sensor_name,
  const std::int32_t state_id) noexcept
{
  auto * vertex = findVertex(graph, sensor_name, state_id);
  return vertex ? vertex->GetObject() : nullptr;
}

karto::LocalizedRangeScan * findScanByUniqueId(
  karto::MapperGraph * graph,
  const std::int32_t unique_id) noexcept
{
  if (!graph) {
    return nullptr;
  }
  for (const auto & sensor_vertices : graph->GetVertices()) {
    for (const auto & state_vertex : sensor_vertices.second) {
      auto * vertex = state_vertex.second;
      if (
        vertex && vertex->GetObject() &&
        vertex->GetObject()->GetUniqueId() == unique_id)
      {
        return vertex->GetObject();
      }
    }
  }
  return nullptr;
}

bool finitePose(const karto::Pose2 & pose) noexcept
{
  return std::isfinite(pose.GetX()) && std::isfinite(pose.GetY()) &&
         std::isfinite(pose.GetHeading());
}

double squaredReferenceDistance(
  const karto::LocalizedRangeScan * first,
  const karto::LocalizedRangeScan * second,
  const bool use_scan_barycenter) noexcept
{
  if (!first || !second) {
    return std::numeric_limits<double>::infinity();
  }
  const karto::Pose2 first_pose = first->GetReferencePose(use_scan_barycenter);
  const karto::Pose2 second_pose = second->GetReferencePose(use_scan_barycenter);
  return first_pose.GetPosition().SquaredDistance(second_pose.GetPosition());
}

karto::LocalizedRangeScan * closestScanToPose(
  const karto::LocalizedRangeScanVector & scans,
  const karto::Pose2 & pose,
  const bool use_scan_barycenter) noexcept
{
  karto::LocalizedRangeScan * closest = nullptr;
  double best_squared_distance = std::numeric_limits<double>::infinity();
  for (auto * scan : scans) {
    if (!scan) {
      continue;
    }
    const karto::Pose2 scan_pose = scan->GetReferencePose(use_scan_barycenter);
    const double squared_distance = pose.GetPosition().SquaredDistance(
      scan_pose.GetPosition());
    if (squared_distance < best_squared_distance) {
      best_squared_distance = squared_distance;
      closest = scan;
    }
  }
  return closest;
}

bool edgeConnects(
  const karto::Edge<karto::LocalizedRangeScan> * edge,
  const std::int32_t first_unique_id,
  const std::int32_t second_unique_id) noexcept
{
  if (
    !edge || !edge->GetSource() || !edge->GetTarget() ||
    !edge->GetSource()->GetObject() || !edge->GetTarget()->GetObject())
  {
    return false;
  }
  const auto source_id = edge->GetSource()->GetObject()->GetUniqueId();
  const auto target_id = edge->GetTarget()->GetObject()->GetUniqueId();
  return
    (source_id == first_unique_id && target_id == second_unique_id) ||
    (source_id == second_unique_id && target_id == first_unique_id);
}

karto::Edge<karto::LocalizedRangeScan> * findConnectingEdge(
  const std::vector<karto::Edge<karto::LocalizedRangeScan> *> & edges,
  const std::int32_t first_unique_id,
  const std::int32_t second_unique_id) noexcept
{
  const auto found = std::find_if(
    edges.begin(), edges.end(),
    [&](const auto * edge) {
      return edgeConnects(edge, first_unique_id, second_unique_id);
    });
  return found == edges.end() ? nullptr : *found;
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

std::vector<karto::LocalizedRangeScan *> nearLinkedScansBeforeProcess(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * current_scan,
  karto::LocalizedRangeScan * previous_scan,
  const karto::LocalizedRangeScanVector & running_scans,
  std::set<std::int32_t> & prelinked_source_unique_ids)
{
  std::vector<karto::LocalizedRangeScan *> scans;
  auto * graph = mapper->GetGraph();
  if (!graph) {
    return scans;
  }

  const bool use_scan_barycenter = mapper->getParamUseScanBarycenter();
  const double maximum_distance = mapper->getParamLinkScanMaximumDistance();
  const double maximum_distance_squared = maximum_distance * maximum_distance;
  const karto::Pose2 center = current_scan->GetReferencePose(use_scan_barycenter);

  std::deque<karto::Vertex<karto::LocalizedRangeScan> *> queue;
  std::set<karto::Vertex<karto::LocalizedRangeScan> *> seen;
  const auto add_seed = [&](karto::LocalizedRangeScan * scan) {
      if (!scan) {
        return;
      }
      prelinked_source_unique_ids.insert(scan->GetUniqueId());
      auto * vertex = findVertex(graph, scan->GetSensorName(), scan->GetStateId());
      if (vertex && seen.insert(vertex).second) {
        queue.push_back(vertex);
      }
    };

  add_seed(previous_scan);

  karto::LocalizedRangeScan * running_closest = closestScanToPose(
    running_scans, center, use_scan_barycenter);
  if (
    running_closest &&
    squaredReferenceDistance(
      current_scan, running_closest, use_scan_barycenter) <
    maximum_distance_squared + kKartoTolerance)
  {
    add_seed(running_closest);
  }

  while (!queue.empty()) {
    auto * vertex = queue.front();
    queue.pop_front();
    if (!vertex || !vertex->GetObject()) {
      continue;
    }
    auto * scan = vertex->GetObject();
    const karto::Pose2 pose = scan->GetReferencePose(use_scan_barycenter);
    const double squared_distance = pose.GetPosition().SquaredDistance(
      center.GetPosition());
    if (squared_distance > maximum_distance_squared - kKartoTolerance) {
      continue;
    }
    scans.push_back(scan);
    for (auto * adjacent : vertex->GetAdjacentVertices()) {
      if (adjacent && seen.insert(adjacent).second) {
        queue.push_back(adjacent);
      }
    }
  }
  return scans;
}

std::vector<karto::LocalizedRangeScanVector> buildNearChainsBeforeProcess(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * current_scan,
  const std::vector<karto::LocalizedRangeScan *> & near_linked_scans,
  const std::int32_t target_state_id)
{
  std::vector<karto::LocalizedRangeScanVector> near_chains;
  auto * graph = mapper->GetGraph();
  if (!graph) {
    return near_chains;
  }

  const bool use_scan_barycenter = mapper->getParamUseScanBarycenter();
  const double maximum_distance = mapper->getParamLinkScanMaximumDistance();
  const double maximum_distance_squared = maximum_distance * maximum_distance;
  const karto::Pose2 scan_pose = current_scan->GetReferencePose(
    use_scan_barycenter);
  std::unordered_set<karto::LocalizedRangeScan *> processed;

  for (auto * near_scan : near_linked_scans) {
    if (!near_scan || near_scan == current_scan || processed.count(near_scan) != 0U) {
      continue;
    }
    processed.insert(near_scan);
    bool valid_chain = true;
    std::list<karto::LocalizedRangeScan *> chain;
    const karto::Name chain_sensor_name = near_scan->GetSensorName();

    for (std::int32_t candidate_id = near_scan->GetStateId() - 1;
      candidate_id >= 0; --candidate_id)
    {
      auto * candidate = findScan(graph, chain_sensor_name, candidate_id);
      if (!candidate) {
        continue;
      }
      const karto::Pose2 candidate_pose = candidate->GetReferencePose(
        use_scan_barycenter);
      const double squared_distance = scan_pose.GetPosition().SquaredDistance(
        candidate_pose.GetPosition());
      if (squared_distance < maximum_distance_squared + kKartoTolerance) {
        chain.push_front(candidate);
        processed.insert(candidate);
      } else {
        break;
      }
    }

    chain.push_back(near_scan);
    const auto sensor_vertices = graph->GetVertices().find(chain_sensor_name);
    const std::int32_t existing_end =
      sensor_vertices == graph->GetVertices().end() ||
      sensor_vertices->second.empty() ? 0 :
      sensor_vertices->second.rbegin()->first + 1;
    const bool same_sensor = chain_sensor_name == current_scan->GetSensorName();
    const std::int32_t end = existing_end + (same_sensor ? 1 : 0);
    for (std::int32_t candidate_id = near_scan->GetStateId() + 1;
      candidate_id < end; ++candidate_id)
    {
      karto::LocalizedRangeScan * candidate = nullptr;
      if (same_sensor && candidate_id == target_state_id) {
        candidate = current_scan;
      } else {
        candidate = findScan(graph, chain_sensor_name, candidate_id);
      }
      if (candidate == current_scan) {
        valid_chain = false;
      }
      if (!candidate) {
        continue;
      }
      const karto::Pose2 candidate_pose = candidate->GetReferencePose(
        use_scan_barycenter);
      const double squared_distance = scan_pose.GetPosition().SquaredDistance(
        candidate_pose.GetPosition());
      if (squared_distance < maximum_distance_squared + kKartoTolerance) {
        chain.push_back(candidate);
        processed.insert(candidate);
      } else {
        break;
      }
    }
    if (valid_chain) {
      near_chains.emplace_back(chain.begin(), chain.end());
    }
  }
  return near_chains;
}

std::pair<std::int32_t, bool> predictTargetStateId(
  karto::MapperGraph * graph,
  const karto::Name & sensor_name) noexcept
{
  if (!graph) {
    return {-1, false};
  }
  const auto sensor_vertices = graph->GetVertices().find(sensor_name);
  if (
    sensor_vertices == graph->GetVertices().end() ||
    sensor_vertices->second.empty())
  {
    return {0, true};
  }
  const auto & vertices = sensor_vertices->second;
  const std::int32_t next = vertices.rbegin()->first + 1;
  return {next, static_cast<std::size_t>(next) == vertices.size()};
}

std::int32_t predictTargetUniqueId(karto::Mapper * mapper) noexcept
{
  std::int32_t maximum = -1;
  for (auto * scan : mapper->GetAllProcessedScans()) {
    if (scan) {
      maximum = std::max(maximum, scan->GetUniqueId());
    }
  }
  return maximum + 1;
}

}  // namespace

NearChainEdgeAccounting evaluateNearChainEdgeAccounting(
  const std::int32_t target_unique_id,
  const std::vector<std::int32_t> & candidate_source_unique_ids,
  const std::vector<std::int32_t> & prelinked_source_unique_ids,
  const std::vector<std::int32_t> & externally_classified_source_unique_ids,
  const std::vector<NearChainEdgeEndpoint> & new_edges) noexcept
{
  NearChainEdgeAccounting accounting;
  if (target_unique_id < 0) {
    return accounting;
  }

  std::unordered_set<std::int32_t> candidate_sources;
  accounting.candidate_endpoints_unique = std::all_of(
    candidate_source_unique_ids.begin(), candidate_source_unique_ids.end(),
    [&](const std::int32_t source_id) {
      return source_id >= 0 && candidate_sources.insert(source_id).second;
    });
  const std::unordered_set<std::int32_t> prelinked_sources(
    prelinked_source_unique_ids.begin(), prelinked_source_unique_ids.end());
  const std::unordered_set<std::int32_t> externally_classified_sources(
    externally_classified_source_unique_ids.begin(),
    externally_classified_source_unique_ids.end());

  for (const auto & edge : new_edges) {
    if (
      !edge.link_info_available || edge.source_unique_id < 0 ||
      edge.target_unique_id < 0 ||
      (edge.source_unique_id != target_unique_id &&
      edge.target_unique_id != target_unique_id))
    {
      ++accounting.unexpected_edge_count;
      continue;
    }
    const std::int32_t other_id =
      edge.source_unique_id == target_unique_id ?
      edge.target_unique_id : edge.source_unique_id;
    if (externally_classified_sources.count(other_id) != 0U) {
      ++accounting.classified_edge_count;
    } else if (prelinked_sources.count(other_id) != 0U) {
      ++accounting.classified_edge_count;
    } else if (candidate_sources.count(other_id) != 0U) {
      ++accounting.classified_edge_count;
      ++accounting.candidate_edge_count;
    } else {
      ++accounting.unexpected_edge_count;
    }
  }
  accounting.complete =
    accounting.candidate_endpoints_unique &&
    accounting.unexpected_edge_count == 0U;
  return accounting;
}

NearChainQualityObservation KartoNearChainCandidate::qualityObservation()
const noexcept
{
  NearChainQualityObservation observation;
  observation.available =
    std::isfinite(normalized_response) &&
    std::isfinite(translation_disagreement_m) &&
    std::isfinite(yaw_disagreement_rad) &&
    std::isfinite(xy_covariance_max_eigenvalue_m2) &&
    std::isfinite(factor_covariance(2, 2));
  observation.normalized_response = normalized_response;
  observation.translation_disagreement_m = translation_disagreement_m;
  observation.yaw_disagreement_rad = yaw_disagreement_rad;
  observation.xy_covariance_max_eigenvalue_m2 =
    xy_covariance_max_eigenvalue_m2;
  observation.yaw_covariance_rad2 = factor_covariance(2, 2);
  return observation;
}

KartoNearChainObservationBatch observeKartoNearChainCandidates(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * current_scan,
  karto::LocalizedRangeScan * previous_scan,
  const karto::LocalizedRangeScanVector & running_scans)
{
  KartoNearChainObservationBatch batch;
  if (!mapper || !current_scan) {
    batch.failure_reason = "missing_mapper_or_scan";
    return batch;
  }
  auto * graph = mapper->GetGraph();
  auto * matcher = mapper->GetSequentialScanMatcher();
  if (!graph || !matcher || !mapper->getParamUseScanMatching()) {
    batch.failure_reason = "karto_observer_dependency_unavailable";
    return batch;
  }
  if (!finitePose(current_scan->GetCorrectedPose())) {
    batch.failure_reason = "current_corrected_pose_invalid";
    return batch;
  }

  const auto target_state = predictTargetStateId(
    graph, current_scan->GetSensorName());
  batch.predicted_target_state_id = target_state.first;
  batch.predicted_target_unique_id = predictTargetUniqueId(mapper);
  batch.target_id_prediction_reliable = target_state.second;
  batch.baseline_response_threshold =
    mapper->getParamLinkMatchMinimumResponseFine();
  batch.applied_response_threshold = batch.baseline_response_threshold;
  if (
    !batch.target_id_prediction_reliable ||
    !std::isfinite(batch.baseline_response_threshold))
  {
    batch.failure_reason = "target_id_or_threshold_unreliable";
    return batch;
  }

  std::set<std::int32_t> prelinked_source_unique_ids;
  const auto near_linked_scans = nearLinkedScansBeforeProcess(
    mapper,
    current_scan,
    previous_scan,
    running_scans,
    prelinked_source_unique_ids);
  batch.prelinked_source_unique_ids.assign(
    prelinked_source_unique_ids.begin(),
    prelinked_source_unique_ids.end());
  const auto near_chains = buildNearChainsBeforeProcess(
    mapper,
    current_scan,
    near_linked_scans,
    batch.predicted_target_state_id);
  const bool use_scan_barycenter = mapper->getParamUseScanBarycenter();
  const double maximum_distance = mapper->getParamLinkScanMaximumDistance();
  const double maximum_distance_squared = maximum_distance * maximum_distance;
  const std::size_t minimum_chain_size = static_cast<std::size_t>(
    std::max(mapper->getParamLoopMatchMinimumChainSize(), 0));
  const karto::Pose2 current_reference = current_scan->GetReferencePose(
    use_scan_barycenter);
  const karto::Pose2 prediction_pose = current_scan->GetCorrectedPose();

  for (const auto & chain : near_chains) {
    KartoNearChainCandidate candidate;
    candidate.covariance.SetToIdentity();
    candidate.factor_covariance.SetToIdentity();
    candidate.candidate_index = batch.candidates.size();
    candidate.sensor_name = current_scan->GetSensorName().ToString();
    candidate.sequential_matched_target_pose = prediction_pose;
    candidate.target_state_id = batch.predicted_target_state_id;
    candidate.target_unique_id = batch.predicted_target_unique_id;
    candidate.chain_size_accepted = chain.size() >= minimum_chain_size;
    for (const auto * scan : chain) {
      if (scan && scan != current_scan) {
        candidate.chain_state_ids.push_back(scan->GetStateId());
        candidate.chain_unique_ids.push_back(scan->GetUniqueId());
      }
    }
    if (!candidate.chain_size_accepted) {
      batch.candidates.push_back(std::move(candidate));
      continue;
    }
    ++batch.eligible_chain_count;

    karto::Pose2 matched_sensor_pose;
    karto::Matrix3 covariance;
    covariance.SetToIdentity();
    candidate.normalized_response = matcher->MatchScan(
      current_scan,
      chain,
      matched_sensor_pose,
      covariance,
      false);
    candidate.matched_sensor_pose = matched_sensor_pose;
    candidate.covariance = covariance;
    auto * source_scan = closestScanToPose(
      chain, current_reference, use_scan_barycenter);
    if (!source_scan) {
      batch.failure_reason = "closest_source_scan_unavailable";
      return batch;
    }
    candidate.source_state_id = source_scan->GetStateId();
    candidate.source_unique_id = source_scan->GetUniqueId();
    candidate.source_distance_m = std::sqrt(
      std::max(
        0.0,
        squaredReferenceDistance(
          current_scan, source_scan, use_scan_barycenter)));
    candidate.distance_accepted =
      candidate.source_distance_m * candidate.source_distance_m <
      maximum_distance_squared + kKartoTolerance;
    candidate.prelinked_by_sequential_or_running =
      prelinked_source_unique_ids.count(candidate.source_unique_id) != 0U;
    candidate.baseline_response_accepted =
      candidate.normalized_response >
      batch.baseline_response_threshold - kKartoTolerance;
    candidate.applied_response_accepted = candidate.baseline_response_accepted;
    candidate.baseline_pose_fusion_used = candidate.baseline_response_accepted;
    if (candidate.baseline_pose_fusion_used) {
      ++batch.baseline_pose_fusion_count;
    }

    const karto::Pose2 matched_target_pose = current_scan->GetCorrectedAt(
      matched_sensor_pose);
    candidate.matched_target_pose = matched_target_pose;
    candidate.translation_disagreement_m = std::hypot(
      matched_target_pose.GetX() - prediction_pose.GetX(),
      matched_target_pose.GetY() - prediction_pose.GetY());
    candidate.yaw_disagreement_rad = std::fabs(
      normalizedAngle(
        matched_target_pose.GetHeading() - prediction_pose.GetHeading()));
    const double xy_trace = covariance(0, 0) + covariance(1, 1);
    const double xy_discriminant = std::sqrt(
      std::max(
        0.0,
        (covariance(0, 0) - covariance(1, 1)) *
        (covariance(0, 0) - covariance(1, 1)) +
        4.0 * covariance(0, 1) * covariance(0, 1)));
    candidate.xy_covariance_max_eigenvalue_m2 =
      0.5 * (xy_trace + xy_discriminant);

    karto::LinkInfo measured_link(
      source_scan->GetCorrectedPose(), matched_target_pose, covariance);
    karto::LinkInfo predicted_link(
      source_scan->GetCorrectedPose(), prediction_pose, covariance);
    candidate.relative_transform = measured_link.GetPoseDifference();
    candidate.factor_covariance = measured_link.GetCovariance();
    setResidual(
      predicted_link.GetPoseDifference(),
      measured_link.GetPoseDifference(),
      candidate.pre_residual_x_m,
      candidate.pre_residual_y_m,
      candidate.pre_residual_yaw_rad);
    candidate.covariance_weighted_disagreement_sq = covarianceWeightedSquared(
      candidate.pre_residual_x_m,
      candidate.pre_residual_y_m,
      candidate.pre_residual_yaw_rad,
      measured_link.GetCovariance());
    batch.candidates.push_back(std::move(candidate));
  }

  std::unordered_set<std::int32_t> candidate_endpoints;
  batch.candidate_endpoints_unique = std::all_of(
    batch.candidates.begin(), batch.candidates.end(),
    [&](const KartoNearChainCandidate & candidate) {
      if (!candidate.chain_size_accepted) {
        return true;
      }
      return candidate.source_unique_id >= 0 &&
      candidate_endpoints.insert(candidate.source_unique_id).second;
    });
  if (!batch.candidate_endpoints_unique) {
    batch.failure_reason = "duplicate_candidate_endpoint";
  }

  batch.available = true;
  return batch;
}

void updateKartoNearChainCandidateOutcomes(
  KartoNearChainObservationBatch & batch,
  karto::Mapper * mapper,
  const karto::LocalizedRangeScan * processed_scan,
  const double applied_response_threshold,
  const std::size_t edge_count_before,
  const std::vector<std::int32_t> & externally_classified_source_unique_ids)
noexcept
{
  batch.applied_response_threshold = applied_response_threshold;
  batch.applied_pose_fusion_count = 0;
  batch.actual_candidate_edge_count = 0;
  batch.new_edge_accounting_complete = false;
  batch.new_edge_count = 0;
  batch.classified_new_edge_count = 0;
  batch.unexpected_new_edge_count = 0;
  batch.outcome_parity = batch.available && mapper && processed_scan;
  if (!batch.outcome_parity || !mapper->GetGraph()) {
    batch.outcome_parity = false;
    return;
  }
  batch.outcome_parity =
    batch.outcome_parity && batch.candidate_endpoints_unique;
  if (
    processed_scan->GetStateId() != batch.predicted_target_state_id ||
    processed_scan->GetUniqueId() != batch.predicted_target_unique_id)
  {
    batch.outcome_parity = false;
    batch.failure_reason = "target_id_prediction_mismatch";
    return;
  }

  const auto & edges = mapper->GetGraph()->GetEdges();
  if (edges.size() < edge_count_before) {
    batch.outcome_parity = false;
    batch.failure_reason = "edge_count_rollback";
    return;
  }
  std::vector<NearChainEdgeEndpoint> new_edge_endpoints;
  new_edge_endpoints.reserve(edges.size() - edge_count_before);
  for (std::size_t index = edge_count_before; index < edges.size(); ++index) {
    auto * edge = edges[index];
    NearChainEdgeEndpoint endpoint;
    if (
      edge && edge->GetSource() && edge->GetTarget() &&
      edge->GetSource()->GetObject() && edge->GetTarget()->GetObject())
    {
      endpoint.source_unique_id =
        edge->GetSource()->GetObject()->GetUniqueId();
      endpoint.target_unique_id =
        edge->GetTarget()->GetObject()->GetUniqueId();
      endpoint.link_info_available =
        dynamic_cast<karto::LinkInfo *>(edge->GetLabel()) != nullptr;
    }
    new_edge_endpoints.push_back(endpoint);
  }
  std::vector<std::int32_t> candidate_source_ids;
  for (const auto & candidate : batch.candidates) {
    if (candidate.chain_size_accepted) {
      candidate_source_ids.push_back(candidate.source_unique_id);
    }
  }
  const NearChainEdgeAccounting accounting = evaluateNearChainEdgeAccounting(
    processed_scan->GetUniqueId(),
    candidate_source_ids,
    batch.prelinked_source_unique_ids,
    externally_classified_source_unique_ids,
    new_edge_endpoints);
  batch.new_edge_count = new_edge_endpoints.size();
  batch.classified_new_edge_count = accounting.classified_edge_count;
  batch.unexpected_new_edge_count = accounting.unexpected_edge_count;
  batch.actual_candidate_edge_count = accounting.candidate_edge_count;
  batch.new_edge_accounting_complete = accounting.complete;
  batch.candidate_endpoints_unique =
    batch.candidate_endpoints_unique && accounting.candidate_endpoints_unique;
  batch.outcome_parity =
    batch.outcome_parity && batch.new_edge_accounting_complete;
  if (!batch.new_edge_accounting_complete && batch.failure_reason.empty()) {
    batch.failure_reason = "unclassified_new_edge";
  }

  for (auto & candidate : batch.candidates) {
    if (!candidate.chain_size_accepted || !std::isfinite(candidate.normalized_response)) {
      candidate.outcome_parity = true;
      continue;
    }
    candidate.applied_response_accepted =
      candidate.normalized_response > applied_response_threshold - kKartoTolerance;
    if (candidate.applied_response_accepted) {
      ++batch.applied_pose_fusion_count;
    }
    auto * actual_edge = findConnectingEdge(
      edges,
      candidate.source_unique_id,
      processed_scan->GetUniqueId());
    candidate.actual_edge_present = actual_edge != nullptr;
    const bool expected_edge_present =
      candidate.prelinked_by_sequential_or_running ||
      (candidate.applied_response_accepted && candidate.distance_accepted);
    bool factor_matches = true;
    auto * actual_link = actual_edge ?
      dynamic_cast<karto::LinkInfo *>(actual_edge->GetLabel()) : nullptr;
    if (
      candidate.actual_edge_present &&
      !candidate.prelinked_by_sequential_or_running)
    {
      factor_matches =
        actual_link && actual_edge->GetSource() && actual_edge->GetTarget() &&
        actual_edge->GetSource()->GetObject() &&
        actual_edge->GetTarget()->GetObject() &&
        actual_edge->GetSource()->GetObject()->GetUniqueId() ==
        candidate.source_unique_id &&
        actual_edge->GetTarget()->GetObject()->GetUniqueId() ==
        processed_scan->GetUniqueId();
      if (factor_matches) {
        const karto::Pose2 & actual_measurement =
          actual_link->GetPoseDifference();
        const karto::Matrix3 & actual_covariance =
          actual_link->GetCovariance();
        const double measurement_error =
          std::hypot(
          actual_measurement.GetX() - candidate.relative_transform.GetX(),
          actual_measurement.GetY() - candidate.relative_transform.GetY()) +
          std::fabs(
          normalizedAngle(
            actual_measurement.GetHeading() -
            candidate.relative_transform.GetHeading()));
        double covariance_error = 0.0;
        for (std::size_t row = 0; row < 3; ++row) {
          for (std::size_t column = 0; column < 3; ++column) {
            covariance_error = std::max(
              covariance_error,
              std::fabs(
                actual_covariance(row, column) -
                candidate.factor_covariance(row, column)));
          }
        }
        factor_matches =
          measurement_error <= 1.0e-5 && covariance_error <= 1.0e-5;
      }
    }
    candidate.outcome_parity =
      candidate.actual_edge_present == expected_edge_present &&
      factor_matches && batch.new_edge_accounting_complete &&
      batch.candidate_endpoints_unique;
    batch.outcome_parity = batch.outcome_parity && candidate.outcome_parity;

    auto * source_scan = findScanByUniqueId(
      mapper->GetGraph(), candidate.source_unique_id);
    if (
      source_scan && actual_link && factor_matches &&
      !candidate.prelinked_by_sequential_or_running)
    {
      karto::Transform source_to_local(
        source_scan->GetCorrectedPose(), karto::Pose2());
      const karto::Pose2 optimized_relative = source_to_local.TransformPose(
        processed_scan->GetCorrectedPose());
      setResidual(
        optimized_relative,
        actual_link->GetPoseDifference(),
        candidate.post_residual_x_m,
        candidate.post_residual_y_m,
        candidate.post_residual_yaw_rad);
      candidate.post_covariance_weighted_residual_sq = covarianceWeightedSquared(
        candidate.post_residual_x_m,
        candidate.post_residual_y_m,
        candidate.post_residual_yaw_rad,
        actual_link->GetCovariance());
    }
  }
}

}  // namespace slam_2d_core
