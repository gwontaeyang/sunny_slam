// Copyright 2026 Gole Robotics

#ifndef SLAM_3D_CORE__INDOOR_TERRAIN_EVIDENCE_HPP_
#define SLAM_3D_CORE__INDOOR_TERRAIN_EVIDENCE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gole
{
namespace slam_3d_core
{

enum class TerrainLabel : std::uint8_t
{
  kUnknown = 0,
  kFloor = 1,
  kRamp = 2,
  kPositiveStep = 3,
  kObservedDrop = 4,
  kStaircase = 5,
};

enum class TerrainSupportState : std::uint8_t
{
  kUnresolved = 0,
  kProbableSupport = 1,
  kConfirmedSupport = 2,
  kProbableHazard = 3,
  kConfirmedHazard = 4,
};

constexpr std::size_t kTerrainLabelCount = 6U;

const char * terrainLabelName(TerrainLabel label);
float terrainLabelValue(TerrainLabel label);
std::array<std::uint8_t, 3U> terrainLabelColor(TerrainLabel label);
const char * terrainSupportStateName(TerrainSupportState state);
std::array<std::uint8_t, 3U> terrainSupportColor(
  TerrainLabel label,
  TerrainSupportState state);
std::optional<Eigen::Matrix3d> gravityAlignmentRotation(
  const Eigen::Vector3d & up_in_source_frame);

struct IndoorTerrainConfig
{
  bool enabled{false};
  int azimuth_bins{120};
  double radial_bin_size_m{0.25};
  double min_range_m{0.8};
  double max_range_m{12.0};
  double seed_max_range_m{3.0};
  double classification_voxel_size_m{0.10};
  int min_points_per_bin{2};
  double floor_quantile{0.20};
  double seed_reference_height_m{0.0};
  double seed_height_cluster_tolerance_m{0.10};
  int seed_min_cluster_bins{3};
  double max_profile_height_above_base_m{0.50};
  double max_seed_height_delta_m{0.45};
  double ground_band_m{0.10};
  double max_adjacent_gap_m{0.55};
  double floor_max_slope_deg{4.0};
  double ramp_min_slope_deg{5.0};
  double ramp_max_slope_deg{22.0};
  int ramp_min_run_bins{3};
  double ramp_min_run_length_m{0.50};
  bool ramp_plane_validation_enabled{false};
  double ramp_plane_neighborhood_radius_m{0.75};
  int ramp_plane_min_points{12};
  double ramp_plane_min_length_m{0.50};
  double ramp_plane_min_width_m{0.20};
  double ramp_plane_max_rmse_m{0.05};
  double ramp_plane_max_outlier_distance_m{0.08};
  double ramp_plane_min_inlier_ratio{0.70};
  bool roughness_preview_enabled{true};
  double roughness_min_height_m{0.025};
  double roughness_max_score{4.0};
  double step_min_height_m{0.08};
  double step_max_height_m{0.30};
  double drop_min_height_m{0.12};
  double drop_max_height_m{0.60};
  int transition_support_bins{1};
  bool staircase_enabled{true};
  int staircase_min_risers{3};
  double staircase_min_riser_height_m{0.11};
  double staircase_max_riser_height_m{0.30};
  double staircase_min_tread_depth_m{0.15};
  double staircase_max_tread_depth_m{0.45};
  double staircase_max_riser_height_deviation_m{0.06};
  double staircase_max_tread_depth_deviation_m{0.15};
  double staircase_min_slope_deg{20.0};
  double staircase_max_slope_deg{55.0};
  double staircase_max_profile_height_above_base_m{2.0};
  bool require_gravity_alignment{true};
};

struct TerrainClassificationStats
{
  bool evaluated{false};
  bool seed_valid{false};
  std::size_t input_points{0U};
  std::size_t profile_bins{0U};
  std::size_t floor_points{0U};
  std::size_t ramp_points{0U};
  std::size_t positive_step_points{0U};
  std::size_t observed_drop_points{0U};
  std::size_t staircase_points{0U};
  std::size_t roughness_preview_points{0U};
  std::size_t ramp_plane_candidate_points{0U};
  std::size_t ramp_plane_validated_points{0U};
  std::size_t ramp_plane_rejected_points{0U};
  std::size_t ramp_plane_fits_attempted{0U};
  std::size_t ramp_plane_fits_accepted{0U};
  std::size_t ramp_plane_rejected_insufficient_points{0U};
  std::size_t ramp_plane_rejected_plane_estimation{0U};
  std::size_t ramp_plane_rejected_inlier_support{0U};
  std::size_t ramp_plane_rejected_length{0U};
  std::size_t ramp_plane_rejected_width{0U};
  std::size_t ramp_plane_rejected_slope{0U};
  std::size_t ramp_plane_rejected_rmse{0U};
  double seed_floor_z_m{0.0};
  double max_roughness_score{0.0};
  double max_accepted_ramp_plane_rmse_m{0.0};
};

struct TerrainClassificationResult
{
  std::vector<TerrainLabel> labels;
  std::vector<double> roughness_scores;
  TerrainClassificationStats stats;
};

TerrainClassificationResult classifyIndoorTerrain(
  const std::vector<Eigen::Vector3d> & points_gravity_aligned,
  const Eigen::Vector3d & sensor_origin,
  const IndoorTerrainConfig & config);

struct TerrainAccumulatorConfig
{
  double voxel_size_m{0.20};
  int min_confirmations{2};
  double min_viewpoint_distance_m{0.30};
  double min_view_label_confidence{0.60};
  double min_label_confidence{0.60};
  double probable_support_min_confidence{0.60};
  double probable_support_max_roughness_score{1.50};
  bool probable_support_require_confirmed_connectivity{true};
  double probable_support_max_anchor_distance_m{0.45};
  double probable_support_max_neighbor_height_delta_m{0.16};
  double ramp_traversability_scale{0.75};
};

struct ConfirmedTerrainPoint
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  TerrainLabel label{TerrainLabel::kUnknown};
  int confirmations{0};
  int observations{0};
  double confidence{0.0};
  double geometric_roughness_score{0.0};
};

struct TerrainEvidencePoint
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  TerrainLabel label{TerrainLabel::kUnknown};
  int confirmations{0};
  int observations{0};
  double confidence{0.0};
  double geometric_roughness_score{0.0};
  bool confirmed{false};
};

struct TerrainSupportPoint
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  TerrainLabel label{TerrainLabel::kUnknown};
  TerrainSupportState state{TerrainSupportState::kUnresolved};
  int confirmations{0};
  int observations{0};
  double label_confidence{0.0};
  double support_probability{0.0};
  double geometric_traversability_score{0.0};
  double geometric_roughness_score{0.0};
  bool connectivity_rejected{false};
};

struct TerrainAccumulatorStats
{
  std::size_t candidate_voxels{0U};
  std::size_t confirmed_voxels{0U};
  std::size_t conflicted_voxels{0U};
  std::size_t ambiguous_view_observations{0U};
  std::size_t probable_support_voxels{0U};
  std::size_t confirmed_support_voxels{0U};
  std::size_t probable_hazard_voxels{0U};
  std::size_t confirmed_hazard_voxels{0U};
  std::size_t unresolved_voxels{0U};
  std::size_t isolated_probable_support_voxels{0U};
  double support_rebuild_runtime_ms{0.0};
  std::array<std::size_t, kTerrainLabelCount> confirmed_by_label{};
};

class TerrainEvidenceAccumulator
{
public:
  explicit TerrainEvidenceAccumulator(
    const TerrainAccumulatorConfig & config = TerrainAccumulatorConfig{});

  void configure(const TerrainAccumulatorConfig & config);
  void clear();
  void addObservation(
    const std::vector<Eigen::Vector3d> & points,
    const std::vector<TerrainLabel> & labels,
    const Eigen::Vector3d & viewpoint);
  void addObservation(
    const std::vector<Eigen::Vector3d> & points,
    const std::vector<TerrainLabel> & labels,
    const std::vector<double> & roughness_scores,
    const Eigen::Vector3d & viewpoint);
  std::vector<TerrainEvidencePoint> evidence() const;
  std::vector<ConfirmedTerrainPoint> confirmedEvidence() const;
  std::vector<TerrainSupportPoint> supportEvidence() const;
  TerrainAccumulatorStats stats() const;

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

  struct LabelEvidence
  {
    Eigen::Vector3d position_sum{Eigen::Vector3d::Zero()};
    double roughness_score_sum{0.0};
    int confirmations{0};
    std::vector<Eigen::Vector3d> viewpoints;
  };

  struct VoxelEvidence
  {
    std::array<LabelEvidence, kTerrainLabelCount> labels;
    LabelEvidence support;
    LabelEvidence hazard;
    std::vector<Eigen::Vector3d> viewpoints;
  };

  VoxelKey voxelKey(const Eigen::Vector3d & point) const;
  TerrainLabel dominantLabel(const VoxelEvidence & evidence) const;
  TerrainLabel confirmedLabel(const VoxelEvidence & evidence) const;
  std::vector<TerrainSupportPoint> buildSupportEvidence() const;
  void rebuildStatsCache();

  TerrainAccumulatorConfig config_;
  std::unordered_map<VoxelKey, VoxelEvidence, VoxelKeyHash> voxels_;
  std::size_t ambiguous_view_observation_count_{0U};
  std::vector<TerrainSupportPoint> cached_support_evidence_;
  TerrainAccumulatorStats cached_stats_;
};

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__INDOOR_TERRAIN_EVIDENCE_HPP_
