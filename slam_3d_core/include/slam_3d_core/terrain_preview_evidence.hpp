// Copyright 2026 Gole Robotics

#ifndef SLAM_3D_CORE__TERRAIN_PREVIEW_EVIDENCE_HPP_
#define SLAM_3D_CORE__TERRAIN_PREVIEW_EVIDENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "slam_3d_core/indoor_terrain_evidence.hpp"

namespace gole
{
namespace slam_3d_core
{

enum class TerrainPreviewState : std::uint8_t
{
  kTentative = 1,
  kGeometryConfirmed = 2,
  kTraversalConfirmed = 3,
  kUnsafeTraversalConflict = 4,
};

enum class TerrainTraversalResponse : std::uint8_t
{
  kUnknown = 0,
  kStable = 1,
  kShock = 2,
};

const char * terrainPreviewStateName(TerrainPreviewState state);
const char * terrainTraversalResponseName(TerrainTraversalResponse response);
float terrainPreviewValue(TerrainLabel label, TerrainPreviewState state);

struct TerrainPreviewConfig
{
  bool enabled{false};
  double voxel_size_m{0.20};
  double association_radius_m{0.55};
  double exit_radius_m{0.75};
  double max_vertical_delta_m{0.40};
  double max_sample_gap_s{0.50};
  double max_swept_segment_length_m{2.0};
  double min_ahead_distance_m{0.75};
  double max_lateral_distance_m{1.0};
  double min_label_confidence{0.67};
  double roughness_score_gate{1.0};
  double ramp_shock_prior{0.25};
  double positive_step_shock_prior{1.0};
  int min_passage_samples{2};
  double min_passage_duration_s{0.05};
};

struct TerrainTraversalSample
{
  std::int64_t stamp_ns{0};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  TerrainTraversalResponse response{TerrainTraversalResponse::kUnknown};
  double roughness_score{0.0};
  bool continuity_break_before{false};
};

struct TerrainPreviewPoint
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  TerrainLabel label{TerrainLabel::kUnknown};
  TerrainPreviewState state{TerrainPreviewState::kTentative};
  int geometry_confirmations{0};
  int geometry_observations{0};
  double geometry_confidence{0.0};
  double geometric_roughness_score{0.0};
  double predicted_shock_score{0.0};
  int traversal_confirmations{0};
  int shock_passages{0};
  int stable_passages{0};
  int unsafe_traversal_conflicts{0};
  double empirical_shock_probability{0.0};
  std::int64_t first_observed_stamp_ns{0};
  std::int64_t first_traversed_stamp_ns{0};
  double first_preview_distance_m{0.0};
  double first_preview_ahead_distance_m{0.0};
  bool first_seen_ahead{false};
  double first_preview_lead_time_s{0.0};
};

struct TerrainPreviewStats
{
  std::size_t landmarks{0U};
  std::size_t tentative_landmarks{0U};
  std::size_t geometry_confirmed_landmarks{0U};
  std::size_t traversal_confirmed_landmarks{0U};
  std::size_t unsafe_traversal_conflict_landmarks{0U};
  std::size_t roughness_preview_landmarks{0U};
  std::size_t ahead_preview_landmarks{0U};
  std::size_t active_passages{0U};
  std::size_t traversal_samples{0U};
  std::size_t finalized_passages{0U};
  std::size_t shock_passages{0U};
  std::size_t stable_passages{0U};
  std::size_t unsafe_traversal_conflicts{0U};
  std::size_t invalid_samples{0U};
  std::size_t continuity_breaks{0U};
  std::size_t previewed_before_traversal_landmarks{0U};
  double mean_preview_lead_time_s{0.0};
  double max_preview_lead_time_s{0.0};
};

struct TerrainTraversalUpdate
{
  bool accepted{false};
  bool evidence_changed{false};
  std::size_t finalized_passages{0U};
};

class TerrainPreviewTracker
{
public:
  explicit TerrainPreviewTracker(
    const TerrainPreviewConfig & config = TerrainPreviewConfig{});

  void configure(const TerrainPreviewConfig & config);
  void clear();
  void updateGeometry(
    const std::vector<TerrainEvidencePoint> & evidence,
    std::int64_t observation_stamp_ns,
    const Eigen::Vector3d & viewpoint,
    const Eigen::Vector3d & view_forward);
  TerrainTraversalUpdate addTraversalSample(const TerrainTraversalSample & sample);
  std::vector<TerrainPreviewPoint> previewEvidence() const;
  TerrainPreviewStats stats() const;

private:
  struct VoxelKey
  {
    std::int64_t x{0};
    std::int64_t y{0};
    std::int64_t z{0};

    bool operator==(const VoxelKey & other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey & key) const;
  };

  struct PassageState
  {
    bool active{false};
    std::int64_t enter_stamp_ns{0};
    std::int64_t last_stamp_ns{0};
    int sample_count{0};
    bool saw_stable{false};
    bool saw_shock{false};
    double peak_roughness_score{0.0};
  };

  struct Landmark
  {
    TerrainEvidencePoint geometry;
    std::int64_t first_observed_stamp_ns{0};
    std::int64_t first_traversed_stamp_ns{0};
    double first_preview_distance_m{0.0};
    double first_preview_ahead_distance_m{0.0};
    bool first_seen_ahead{false};
    double first_preview_lead_time_s{0.0};
    int traversal_confirmations{0};
    int shock_passages{0};
    int stable_passages{0};
    int unsafe_traversal_conflicts{0};
    PassageState passage;
  };

  VoxelKey voxelKey(const Eigen::Vector3d & point) const;
  bool isUnsafeForTraversal(TerrainLabel label) const;
  double predictedShockScore(const Landmark & landmark) const;
  TerrainPreviewState previewState(const Landmark & landmark) const;
  void resetTraversalEvidence(Landmark & landmark) const;
  bool updateLandmarkPassage(
    Landmark & landmark,
    const TerrainTraversalSample & sample,
    const TerrainTraversalSample * previous_sample,
    std::size_t & finalized_passages) const;
  bool finalizePassage(Landmark & landmark) const;
  bool segmentIsUsable(
    const TerrainTraversalSample & sample,
    const TerrainTraversalSample * previous_sample) const;
  std::unordered_set<VoxelKey, VoxelKeyHash> nearbyLandmarkKeys(
    const TerrainTraversalSample & sample,
    const TerrainTraversalSample * previous_sample) const;
  void rebuildStatsCache();

  TerrainPreviewConfig config_;
  std::unordered_map<VoxelKey, Landmark, VoxelKeyHash> landmarks_;
  std::unordered_set<VoxelKey, VoxelKeyHash> active_passage_keys_;
  std::optional<TerrainTraversalSample> previous_traversal_sample_;
  std::size_t traversal_sample_count_{0U};
  std::size_t invalid_sample_count_{0U};
  std::size_t continuity_break_count_{0U};
  TerrainPreviewStats cached_landmark_stats_;
};

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__TERRAIN_PREVIEW_EVIDENCE_HPP_
