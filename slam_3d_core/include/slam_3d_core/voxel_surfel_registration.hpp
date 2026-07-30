// Copyright 2026 Gole Robotics

#ifndef SLAM_3D_CORE__VOXEL_SURFEL_REGISTRATION_HPP_
#define SLAM_3D_CORE__VOXEL_SURFEL_REGISTRATION_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <array>
#include <cstddef>
#include <string>

namespace gole
{
namespace slam_3d_core
{

class VoxelSurfelRegistration
{
public:
  using PointT = pcl::PointXYZ;
  using CloudT = pcl::PointCloud<PointT>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using RowVector6d = Eigen::Matrix<double, 1, 6>;

  struct Config
  {
    double voxel_size{0.6};
    std::size_t min_points_per_voxel{8};
    double max_planarity_ratio{0.15};
    double min_tangent_variance{1.0e-4};
    double max_plane_distance{0.35};
    double max_center_distance{1.2};

    std::size_t min_correspondences{40};
    double min_correspondence_ratio{0.20};
    double huber_delta{0.10};

    int max_iterations{20};
    int max_backtracking_steps{8};
    double backtracking_factor{0.5};
    double rotation_convergence_rad{1.0e-4};
    double translation_convergence_m{1.0e-4};
    double relative_cost_convergence{1.0e-6};
    double max_rotation_step_rad{0.20};
    double max_translation_step_m{0.50};

    // Normalized state is [rotation_scale * dtheta, translation].
    double rotation_scale{1.0};
    double relative_eigen_threshold{1.0e-3};
    int min_observable_rank{3};

    double max_quality_rmse{0.25};
    double max_quality_p95{0.45};
  };

  struct Result
  {
    Eigen::Isometry3d target_T_source{Eigen::Isometry3d::Identity()};
    bool accepted{false};
    bool converged{false};
    std::string reason{"not_run"};

    std::size_t valid_source_points{0};
    std::size_t target_voxels{0};
    std::size_t planar_surfels{0};
    std::size_t correspondences{0};
    double correspondence_ratio{0.0};

    double rmse{0.0};
    double median_abs_residual{0.0};
    double p95_abs_residual{0.0};
    // Mean Huber loss over the current correspondences.
    double robust_cost{0.0};
    double robust_downweight_ratio{0.0};

    int iterations{0};
    // Eigenvalues are ascending and refer to the rotation-scale-normalized Hessian.
    std::array<double, 6> hessian_eigenvalues{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    int observable_rank{0};
    double hessian_condition{0.0};

    // Diagonal of the weak-subspace projector in [rx, ry, rz, tx, ty, tz] order.
    Vector6d weak_state_vector{Vector6d::Zero()};
    Eigen::Vector3d weak_rotation_axes{Eigen::Vector3d::Zero()};
    Eigen::Vector3d weak_translation_axes{Eigen::Vector3d::Zero()};

    double target_build_time_ms{0.0};
    double optimization_time_ms{0.0};
    double total_time_ms{0.0};
  };

  VoxelSurfelRegistration();
  explicit VoxelSurfelRegistration(const Config & config);

  Result align(
    const CloudT & source,
    const CloudT & target,
    const Eigen::Isometry3d & initial_guess) const;

  const Config & config() const;

  // Increment order is [rotation, translation]:
  // R_new = Exp(dtheta) * R, followed by t_new = t + dt.
  static Eigen::Isometry3d applyIncrement(
    const Eigen::Isometry3d & target_T_source,
    const Vector6d & increment);

  static double pointToPlaneResidual(
    const Eigen::Vector3d & source_point,
    const Eigen::Isometry3d & target_T_source,
    const Eigen::Vector3d & target_centroid,
    const Eigen::Vector3d & target_normal);

  static RowVector6d pointToPlaneJacobian(
    const Eigen::Vector3d & source_point,
    const Eigen::Isometry3d & target_T_source,
    const Eigen::Vector3d & target_normal);

private:
  Config config_;
};

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__VOXEL_SURFEL_REGISTRATION_HPP_
