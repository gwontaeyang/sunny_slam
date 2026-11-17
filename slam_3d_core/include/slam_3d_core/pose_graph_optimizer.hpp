// Copyright 2026 Gole Robotics

#ifndef SLAM_3D_CORE__POSE_GRAPH_OPTIMIZER_HPP_
#define SLAM_3D_CORE__POSE_GRAPH_OPTIMIZER_HPP_

#include <limits>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace gole
{
namespace slam_3d_core
{

struct PoseGraphConstraint
{
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  int from_index{0};
  int to_index{0};
  Eigen::Isometry3d from_T_to{Eigen::Isometry3d::Identity()};
  // Full information matrix in [x, y, z, rx, ry, rz] residual order.
  Matrix6d information_matrix{Matrix6d::Identity()};
  bool loop_closure{false};
};

struct PoseGraphOptimizationConfig
{
  int max_iterations_per_stage{20};
  std::vector<double> robust_scale_factors{4.0, 2.0, 1.0};
  bool robust_loop_enabled{true};
  double loop_cauchy_scale{1.0};
  bool dcs_enabled{true};
  int dcs_start_stage{0};
  double dcs_phi{5.0};
  double dcs_min_scale{0.05};
  double function_tolerance{1.0e-6};
  double gradient_tolerance{1.0e-10};
  double parameter_tolerance{1.0e-8};
  int num_threads{1};
  bool use_nonmonotonic_steps{false};
};

struct PoseGraphOptimizationStageStats
{
  int stage_index{0};
  int iterations{0};
  int successful_steps{0};
  int unsuccessful_steps{0};
  int dcs_downweighted_loops{0};
  double robust_scale_factor{1.0};
  double cauchy_scale{1.0};
  double min_dcs_scale{1.0};
  double mean_dcs_scale{1.0};
  double initial_cost{std::numeric_limits<double>::infinity()};
  double final_cost{std::numeric_limits<double>::infinity()};
  double final_gradient_max_norm{std::numeric_limits<double>::infinity()};
  double runtime_ms{0.0};
  std::string termination_type{"not_run"};
  std::string brief_report;
};

struct PoseGraphOptimizationResult
{
  bool success{false};
  std::vector<Eigen::Isometry3d> poses;
  std::vector<double> final_dcs_scales;
  std::vector<PoseGraphOptimizationStageStats> stages;
  int total_iterations{0};
  int successful_steps{0};
  int unsuccessful_steps{0};
  int dcs_downweighted_loops{0};
  double min_dcs_scale{1.0};
  double initial_cost{std::numeric_limits<double>::infinity()};
  double final_cost{std::numeric_limits<double>::infinity()};
  double final_gradient_max_norm{std::numeric_limits<double>::infinity()};
  double runtime_ms{0.0};
  std::string termination_type{"not_run"};
  std::string failure_reason;
};

bool validPoseGraphOptimizationConfig(
  const PoseGraphOptimizationConfig & config) noexcept;

double poseGraphConstraintChiSquared(
  const PoseGraphConstraint & constraint,
  const std::vector<Eigen::Isometry3d> & poses) noexcept;

double dynamicCovarianceScale(double chi_squared, double phi) noexcept;

class PoseGraphOptimizer
{
public:
  explicit PoseGraphOptimizer(PoseGraphOptimizationConfig config = {});

  const PoseGraphOptimizationConfig & config() const noexcept;
  void setConfig(const PoseGraphOptimizationConfig & config);

  PoseGraphOptimizationResult optimize(
    const std::vector<Eigen::Isometry3d> & initial_poses,
    const std::vector<PoseGraphConstraint> & constraints) const;

private:
  PoseGraphOptimizationConfig config_;
};

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__POSE_GRAPH_OPTIMIZER_HPP_
