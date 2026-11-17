#ifndef SLAM_3D_CORE__TEMPORAL_STATIC_MAP_HPP_
#define SLAM_3D_CORE__TEMPORAL_STATIC_MAP_HPP_

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

namespace gole
{
namespace slam_3d_core
{

struct TemporalStaticMapConfig
{
  bool enabled{false};
  double voxel_size_m{0.25};
  int min_hit_observations{3};
  double min_observation_span_s{1.0};
  double min_viewpoint_distance_m{0.60};
  bool ray_clearing_enabled{true};
  double ray_endpoint_margin_m{0.45};
  double max_ray_length_m{30.0};
  int max_rays_per_observation{2000};
  bool prior_free_enabled{false};
  int min_prior_free_observations{4};
  int max_prior_free_observations{8};
  double prior_free_max_sensor_vertical_drop_m{0.0};
  double promotion_min_hit_ratio{0.35};
  double retention_min_hit_ratio{0.20};
  int min_miss_observations_for_demotion{4};
};

bool validTemporalStaticMapConfig(const TemporalStaticMapConfig & config);

enum class TemporalStaticMapSelection
{
  kConfirmed,
  kRejected,
  kAll
};

struct TemporalStaticMapPoint
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  int hit_observations{0};
  int miss_observations{0};
  double observation_span_s{0.0};
  double viewpoint_distance_m{0.0};
  double hit_ratio{0.0};
  bool confirmed{false};
};

struct TemporalStaticMapSensorView
{
  Eigen::Vector3d sensor_origin{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> endpoints;
};

struct TemporalStaticMapStats
{
  std::size_t observations{0U};
  std::size_t input_points{0U};
  std::size_t candidate_voxels{0U};
  std::size_t confirmed_voxels{0U};
  std::size_t rejected_voxels{0U};
  std::size_t contradicted_voxels{0U};
  std::size_t endpoint_hit_updates{0U};
  std::size_t ray_voxel_checks{0U};
  std::size_t ray_miss_updates{0U};
  std::size_t prior_free_voxel_count{0U};
  std::size_t prior_free_update_count{0U};
  std::size_t prior_free_seeded_candidate_count{0U};
  std::size_t promotions{0U};
  std::size_t demotions{0U};
  double last_update_runtime_ms{0.0};
  double cumulative_update_runtime_ms{0.0};
};

class TemporalStaticMapAccumulator
{
public:
  explicit TemporalStaticMapAccumulator(
    const TemporalStaticMapConfig & config = TemporalStaticMapConfig{});

  void setConfig(const TemporalStaticMapConfig & config);
  void reset();
  void addObservation(
    double stamp_s,
    const Eigen::Vector3d & sensor_origin,
    const std::vector<Eigen::Vector3d> & endpoints);
  void addObservation(
    double stamp_s,
    const std::vector<TemporalStaticMapSensorView> & sensor_views);

  std::vector<TemporalStaticMapPoint> points(
    TemporalStaticMapSelection selection) const;
  const TemporalStaticMapStats & stats() const;
  const TemporalStaticMapConfig & config() const;

private:
  struct VoxelKey
  {
    std::int64_t x{0};
    std::int64_t y{0};
    std::int64_t z{0};

    bool operator==(const VoxelKey & other) const;
    bool operator<(const VoxelKey & other) const;
  };

  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey & key) const;
  };

  struct LocalEndpoint
  {
    Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
    Eigen::Vector3d viewpoint_sum{Eigen::Vector3d::Zero()};
    int point_count{0};
    int viewpoint_count{0};
  };

  struct PriorFreeEvidence
  {
    int observations{0};
    std::size_t last_observation{0U};
  };

  struct VoxelEvidence
  {
    Eigen::Vector3d position_sum{Eigen::Vector3d::Zero()};
    Eigen::Vector3d first_viewpoint{Eigen::Vector3d::Zero()};
    int hit_observations{0};
    int miss_observations{0};
    int misses_since_last_hit{0};
    std::size_t last_miss_observation{0U};
    double first_stamp_s{0.0};
    double last_stamp_s{0.0};
    double max_viewpoint_distance_m{0.0};
    bool confirmed{false};
    bool contradicted{false};
    bool have_first_viewpoint{false};
  };

  using VoxelMap = std::unordered_map<VoxelKey, VoxelEvidence, VoxelKeyHash>;
  using LocalEndpointMap = std::unordered_map<VoxelKey, LocalEndpoint, VoxelKeyHash>;
  using PriorFreeMap = std::unordered_map<VoxelKey, PriorFreeEvidence, VoxelKeyHash>;

  VoxelKey voxelKey(const Eigen::Vector3d & point) const;
  double hitRatio(const VoxelEvidence & evidence) const;
  bool matureForPromotion(const VoxelEvidence & evidence) const;
  void updateEvidenceState(VoxelEvidence & evidence);
  void applyRayMisses(
    const Eigen::Vector3d & sensor_origin,
    const Eigen::Vector3d & endpoint,
    const LocalEndpointMap & endpoints_in_observation,
    std::size_t observation_id);

  TemporalStaticMapConfig config_;
  VoxelMap voxels_;
  PriorFreeMap prior_free_voxels_;
  TemporalStaticMapStats stats_;
};

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__TEMPORAL_STATIC_MAP_HPP_
