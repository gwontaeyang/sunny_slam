// Copyright 2026 Gole Robotics

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "slam_3d_core/pose_graph_optimizer.hpp"

namespace
{
using namespace gole::slam_3d_core;  // NOLINT(build/namespaces)
using Matrix6d = Eigen::Matrix<double, 6, 6>;

Eigen::Isometry3d pose(double x, double y, double z, double roll, double pitch, double yaw)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() = Eigen::Vector3d(x, y, z);
  result.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
  return result;
}

PoseGraphConstraint edge(int from, int to, const Eigen::Isometry3d & measurement, bool loop = false)
{
  PoseGraphConstraint result;
  result.from_index = from;
  result.to_index = to;
  result.from_T_to = measurement;
  result.loop_closure = loop;
  Matrix6d root = Matrix6d::Identity();
  root.diagonal() << 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
  root(0, 4) = 0.7;
  root(1, 5) = -0.8;
  root(2, 3) = 1.1;
  result.information_matrix = root.transpose() * root;
  return result;
}

PoseGraphOptimizationConfig configFor(const std::string & backend)
{
  PoseGraphOptimizationConfig config;
  config.backend = backend;
  config.max_iterations_per_stage = 50;
  config.function_tolerance = 1.0e-10;
  config.gradient_tolerance = 1.0e-12;
  config.parameter_tolerance = 1.0e-12;
  return config;
}

class PoseGraphBackend : public ::testing::TestWithParam<std::string> {};

TEST_P(PoseGraphBackend, SolvesAllSixDofsWithRotatedGaugeAndReversedGaugeEdge)
{
  const auto gauge = pose(6, -3, 1, 0.2, -0.3, 2.8);
  const auto target = gauge * pose(1, 2, 0.6, -0.3, 0.2, 0.7);
  const auto third = target * pose(0.5, -0.2, 0.3, 0.1, -0.2, 0.1);
  std::vector<Eigen::Isometry3d> initial{
    gauge, target * pose(0.1, -0.2, 0.3, 0.09, 0.1, -0.1),
    third * pose(-0.2, 0.1, -0.3, -0.1, 0.1, 0.08)};
  auto config = configFor(GetParam());
  const auto result = PoseGraphOptimizer(config).optimize(
    initial, {edge(1, 0, target.inverse() * gauge),
      edge(1, 2, target.inverse() * third), edge(0, 2, gauge.inverse() * third, true)});
  ASSERT_TRUE(result.success) << result.failure_reason;
  EXPECT_EQ(result.backend, GetParam());
  EXPECT_TRUE(result.poses.front().matrix().isApprox(gauge.matrix(), 1.0e-12));
  EXPECT_TRUE(result.poses[1].matrix().isApprox(target.matrix(), 2.0e-6));
  EXPECT_TRUE(result.poses[2].matrix().isApprox(third.matrix(), 2.0e-6));
  EXPECT_LT(result.final_cost, 1.0e-9);
  ASSERT_EQ(result.stages.size(), 3U);
  for (const auto & stage : result.stages) {
    EXPECT_LE(stage.final_cost, stage.initial_cost + 1.0e-12);
    EXPECT_NE(stage.termination_type, "not_run");
    EXPECT_FALSE(stage.brief_report.empty());
    EXPECT_GE(stage.runtime_ms, 0.0);
  }
}

TEST_P(PoseGraphBackend, FullInformationAgreesWithKnownLinearSolution)
{
  std::vector<Eigen::Isometry3d> initial{
    Eigen::Isometry3d::Identity(), pose(0.5, 1, 0, 0, 0, 0)};
  auto first = edge(0, 1, Eigen::Isometry3d::Identity());
  auto second = edge(0, 1, pose(1, 2, 0, 0, 0, 0));
  first.information_matrix.setIdentity();
  second.information_matrix.setIdentity();
  second.information_matrix(0, 0) = 2;
  second.information_matrix(1, 1) = 3;
  second.information_matrix(0, 1) = second.information_matrix(1, 0) = 0.5;
  auto config = configFor(GetParam());
  config.robust_loop_enabled = false;
  config.dcs_enabled = false;
  config.robust_scale_factors = {1.0};
  const auto result = PoseGraphOptimizer(config).optimize(initial, {first, second});
  ASSERT_TRUE(result.success) << result.failure_reason;
  EXPECT_NEAR(result.poses[1].translation().x(), 8.75 / 11.75, 2.0e-6);
  EXPECT_NEAR(result.poses[1].translation().y(), 18.0 / 11.75, 2.0e-6);
}

TEST_P(PoseGraphBackend, RankDeficientFactorWorksWhenGraphIsObservable)
{
  const auto target = pose(1, -0.5, 0.3, 0.1, -0.1, 0.2);
  auto planar = edge(0, 1, target);
  planar.information_matrix.setZero();
  planar.information_matrix(0, 0) = 10;
  planar.information_matrix(1, 1) = 10;
  auto remaining = edge(0, 1, target);
  remaining.information_matrix.setIdentity();
  remaining.information_matrix(0, 0) = remaining.information_matrix(1, 1) = 0;
  const auto result = PoseGraphOptimizer(configFor(GetParam())).optimize(
    {Eigen::Isometry3d::Identity(), pose(1.1, -0.3, 0.5, 0.2, 0, 0.1)}, {planar, remaining});
  ASSERT_TRUE(result.success) << result.failure_reason;
  EXPECT_TRUE(result.poses[1].matrix().isApprox(target.matrix(), 2.0e-6));
}

TEST_P(PoseGraphBackend, InvalidGraphIsRejectedWithoutChangingInput)
{
  const std::vector<Eigen::Isometry3d> initial{
    Eigen::Isometry3d::Identity(), pose(1, 0, 0, 0, 0, 0), pose(2, 0, 0, 0, 0, 0)};
  auto config = configFor(GetParam());
  auto invalid = edge(0, 1, initial[1]);
  invalid.information_matrix(0, 0) = -10;
  for (const auto & constraints : std::vector<std::vector<PoseGraphConstraint>>{
      {invalid}, {edge(0, 1, initial[1])}, {edge(-1, 1, initial[1])},
      {edge(0, 3, initial[1])}, {edge(1, 1, initial[1])}})
  {
    const auto result = PoseGraphOptimizer(config).optimize(initial, constraints);
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.stages.empty());
    ASSERT_EQ(result.poses.size(), initial.size());
    for (std::size_t index = 0; index < initial.size(); ++index) {
      EXPECT_EQ(result.poses[index].matrix(), initial[index].matrix());
    }
  }
  auto nonfinite = initial;
  nonfinite[1].translation().x() = std::numeric_limits<double>::quiet_NaN();
  const auto result = PoseGraphOptimizer(config).optimize(
    nonfinite, {edge(0, 1, initial[1]), edge(1, 2, initial[1])});
  EXPECT_EQ(result.failure_reason, "nonfinite_initial_pose");
}

TEST_P(PoseGraphBackend, RobustDcsRejectsFalseLoopWithoutWeakeningOdometry)
{
  std::vector<Eigen::Isometry3d> initial;
  std::vector<PoseGraphConstraint> constraints;
  for (int i = 0; i <= 12; ++i) {
    initial.push_back(pose(i, 0, 0, 0, 0, 0));
    if (i > 0) {
      constraints.push_back(edge(i - 1, i, pose(1, 0, 0, 0, 0, 0)));
    }
  }
  constraints.push_back(edge(0, 12, pose(12, 10, 2, 0.1, 0.2, 0.3), true));
  auto config = configFor(GetParam());
  config.dcs_phi = 1;
  config.dcs_min_scale = 0.02;
  const auto result = PoseGraphOptimizer(config).optimize(initial, constraints);
  ASSERT_TRUE(result.success) << result.failure_reason;
  EXPECT_EQ(result.dcs_downweighted_loops, 1);
  EXPECT_LT(result.min_dcs_scale, 0.05);
  const double robust_displacement =
    (result.poses.back().translation() - initial.back().translation()).norm();
  const double false_closure_displacement =
    (constraints.back().from_T_to.translation() - initial.back().translation()).norm();
  EXPECT_LT(robust_displacement, 0.05 * false_closure_displacement);
  config.dcs_enabled = config.robust_loop_enabled = false;
  const auto unprotected = PoseGraphOptimizer(config).optimize(initial, constraints);
  ASSERT_TRUE(unprotected.success) << unprotected.failure_reason;
  EXPECT_LT(
    robust_displacement, 0.1 *
    (unprotected.poses.back().translation() - initial.back().translation()).norm());
  for (std::size_t i = 0; i + 1 < constraints.size(); ++i) {
    EXPECT_DOUBLE_EQ(result.final_dcs_scales[i], 1.0);
  }
}

TEST_P(PoseGraphBackend, QuaternionHemisphereAndMixedInformationHaveConsistentCost)
{
  const auto from = pose(2, 1, 0.3, 0, 0, 2.96705972839036);
  const auto to = pose(1, 2, 0.1, 0, 0, -2.96705972839036);
  const auto measurement = pose(0.2, 0.3, 0.1, 0, 0, 0.2);
  const auto constraint = edge(0, 1, measurement);
  const std::vector<Eigen::Isometry3d> initial{from, to};
  auto config = configFor(GetParam());
  config.robust_loop_enabled = config.dcs_enabled = false;
  config.robust_scale_factors = {1.0};
  const auto result = PoseGraphOptimizer(config).optimize(initial, {constraint});
  ASSERT_TRUE(result.success) << result.failure_reason;
  EXPECT_NEAR(result.initial_cost, 0.5 * poseGraphConstraintChiSquared(constraint, initial), 1e-9);
  EXPECT_TRUE(result.poses[1].matrix().isApprox((from * measurement).matrix(), 2.0e-6));
}

TEST(PoseGraphBackendAvailability, UnknownAndDisabledBackendsNeverFallBack)
{
  EXPECT_TRUE(poseGraphSolverAvailable("ceres_se3"));
  EXPECT_FALSE(poseGraphSolverAvailable("gtsam"));
  auto invalid = configFor("misspelled");
  EXPECT_THROW(PoseGraphOptimizer{invalid}, std::invalid_argument);
  for (const auto & backend : {"g2o_se3", "gtsam_se3"}) {
    if (!poseGraphSolverAvailable(backend)) {
      const std::vector<Eigen::Isometry3d> initial{
        Eigen::Isometry3d::Identity(), pose(1, 0, 0, 0, 0, 0)};
      const auto result = PoseGraphOptimizer(configFor(backend)).optimize(
        initial, {edge(0, 1, initial[1])});
      EXPECT_FALSE(result.success);
      EXPECT_EQ(result.failure_reason, std::string("backend_unavailable:") + backend);
      EXPECT_EQ(result.poses[1].matrix(), initial[1].matrix());
    }
  }
}

TEST(PoseGraphBackendDiagnostics, FailedStageRetainsNativeCostWhilePosesRollBack)
{
  if (!poseGraphSolverAvailable("gtsam_se3")) {
    GTEST_SKIP() << "Requires the GTSAM damping-exhaustion fixture";
  }
  const auto from = pose(2, 1, 0.3, 0, 0, 2.96705972839036);
  const auto to = pose(1, 2, 0.1, 0, 0, -2.96705972839036);
  const auto measurement = pose(0.2, 0.3, 0.1, 0, 0, 0.2);
  const std::vector<Eigen::Isometry3d> initial{from, to};
  auto config = configFor("gtsam_se3");
  config.robust_loop_enabled = config.dcs_enabled = false;
  config.robust_scale_factors = {1.0};
  config.gradient_tolerance = config.function_tolerance = 0.0;
  const auto result = PoseGraphOptimizer(config).optimize(initial, {edge(0, 1, measurement)});
  if (result.success) {
    GTEST_SKIP() <<
      "This library/platform reached exact stationarity instead of damping exhaustion";
  }
  ASSERT_EQ(result.stages.size(), 1U);
  EXPECT_NE(result.failure_reason.find("damping range"), std::string::npos);
  EXPECT_EQ(result.poses[0].matrix(), initial[0].matrix());
  EXPECT_EQ(result.poses[1].matrix(), initial[1].matrix());
  EXPECT_GT(result.stages[0].iterations, 0);
  EXPECT_TRUE(result.stages[0].gradient_available);
  EXPECT_LT(result.final_cost, 1.0e-10 * result.initial_cost);
  EXPECT_EQ(result.final_cost, result.stages[0].final_cost);
  EXPECT_GT(result.runtime_ms, 0.0);
}

struct RampGraph
{
  std::vector<Eigen::Isometry3d> truth, initial;
  std::vector<PoseGraphConstraint> constraints;
};

RampGraph rampGraph()
{
  RampGraph graph;
  constexpr int count = 81;
  for (int i = 0; i < count; ++i) {
    const double t = static_cast<double>(i) / (count - 1);
    const double angle = t * 2.0 * 3.141592653589793;
    graph.truth.push_back(
      pose(
        12 * std::sin(angle), 8 * (1 - std::cos(angle)), 0.7 * std::sin(angle),
        0.03 * std::sin(2 * angle), 0.05 * std::cos(angle), angle));
    graph.initial.push_back(
      graph.truth.back() * pose(
        0.4 * t, -0.2 * t, 0.9 * t, 0.04 * t, -0.06 * t, 0.05 * t));
    if (i > 0) {
      graph.constraints.push_back(
        edge(
          i - 1, i, graph.initial[i - 1].inverse() * graph.initial[i]));
    }
  }
  auto closure = edge(0, count - 1, graph.truth.front().inverse() * graph.truth.back(), true);
  closure.information_matrix *= 5.0;
  graph.constraints.push_back(closure);
  return graph;
}

void writeEvidence(
  const RampGraph & graph, const std::string & backend, const PoseGraphOptimizationResult & result)
{
  const char * directory = std::getenv("SLAM_3D_BACKEND_EVIDENCE_DIR");
  if (!directory) {
    return;
  }
  std::filesystem::create_directories(directory);
  std::ofstream trajectory(std::filesystem::path(directory) / (backend + "_trajectory.csv"));
  trajectory << std::setprecision(17) << "node,source,x_m,y_m,z_m,qx,qy,qz,qw\n";
  const std::vector<std::pair<std::string, const std::vector<Eigen::Isometry3d> *>> sources{
    {"reference", &graph.truth}, {"initial", &graph.initial}, {"optimized", &result.poses}};
  for (const auto & source : sources) {
    for (std::size_t i = 0; i < source.second->size(); ++i) {
      const auto & p = (*source.second)[i];
      const Eigen::Quaterniond q(p.linear());
      trajectory << i << ',' << source.first << ',' << p.translation().x() << ',' <<
        p.translation().y() << ',' << p.translation().z() << ',' << q.x() << ',' <<
        q.y() << ',' << q.z() << ',' << q.w() << '\n';
    }
  }
  std::ofstream stats(std::filesystem::path(directory) / (backend + "_stages.csv"));
  stats << std::setprecision(17) <<
    "stage,initial_cost,final_cost,iterations,runtime_ms,termination,gradient_available,"
    "step_counts_available,min_dcs_scale\n";
  for (const auto & stage : result.stages) {
    stats << stage.stage_index << ',' << stage.initial_cost << ',' << stage.final_cost << ',' <<
      stage.iterations << ',' << stage.runtime_ms << ',' << stage.termination_type << ',' <<
      stage.gradient_available << ',' << stage.step_counts_available << ',' <<
      stage.min_dcs_scale << '\n';
  }
  std::ofstream residuals(std::filesystem::path(directory) / (backend + "_residuals.csv"));
  residuals << std::setprecision(17) << "edge,type,initial_chi2,final_chi2\n";
  for (std::size_t i = 0; i < graph.constraints.size(); ++i) {
    residuals << i << ',' << (graph.constraints[i].loop_closure ? "loop" : "odometry") << ',' <<
      poseGraphConstraintChiSquared(graph.constraints[i], graph.initial) << ',' <<
      poseGraphConstraintChiSquared(graph.constraints[i], result.poses) << '\n';
  }
  ASSERT_TRUE(trajectory.good());
  ASSERT_TRUE(stats.good());
  ASSERT_TRUE(residuals.good());
}

TEST(PoseGraphBackendComparison, SyntheticRampLoopAgreesAcrossSolvers)
{
  const auto graph = rampGraph();
  PoseGraphOptimizationResult baseline;
  for (const auto & backend : availablePoseGraphSolvers()) {
    auto config = configFor(backend);
    config.dcs_start_stage = 1;
    const auto result = PoseGraphOptimizer(config).optimize(graph.initial, graph.constraints);
    ASSERT_TRUE(result.success) << backend << ": " << result.failure_reason;
    EXPECT_LT(poseGraphConstraintChiSquared(graph.constraints.back(), result.poses), 0.1);
    EXPECT_LT(std::abs(result.poses.back().translation().z()), 0.08);
    if (backend == "ceres_se3") {
      baseline = result;
    } else {
      ASSERT_EQ(result.poses.size(), baseline.poses.size());
      EXPECT_NEAR(result.final_cost, baseline.final_cost, 1.0e-5);
      for (std::size_t i = 0; i < result.poses.size(); ++i) {
        EXPECT_LT((result.poses[i].translation() - baseline.poses[i].translation()).norm(), 2.0e-3);
        EXPECT_LT(
          Eigen::AngleAxisd(
            baseline.poses[i].linear().transpose() * result.poses[i].linear()).angle(), 2.0e-4);
      }
    }
    writeEvidence(graph, backend, result);
  }
}

TEST(PoseGraphBackendComparison, MinimalCpuScheduleRunsEachNativeSolverWithinBudget)
{
  const auto graph = rampGraph();
  for (const auto & backend : availablePoseGraphSolvers()) {
    auto config = configFor(backend);
    config.max_iterations_per_stage = 4;
    config.robust_scale_factors = {1.0};
    config.dcs_start_stage = 0;
    const auto result = PoseGraphOptimizer(config).optimize(graph.initial, graph.constraints);
    ASSERT_TRUE(result.success) << backend << ": " << result.failure_reason;
    ASSERT_EQ(result.stages.size(), 1U);
    EXPECT_LT(result.final_cost, result.initial_cost);
    const auto & stage = result.stages.front();
    EXPECT_NE(stage.termination_type, "not_run");
    EXPECT_GT(stage.iterations, 0);
    // Ceres also reports iteration zero (initial evaluation).
    EXPECT_LE(stage.iterations, backend == "ceres_se3" ? 5 : 4);
    for (const auto & optimized_pose : result.poses) {
      EXPECT_TRUE(optimized_pose.matrix().allFinite());
    }
    writeEvidence(graph, backend + "_cpu_minimal", result);
  }
}

INSTANTIATE_TEST_SUITE_P(
  CompiledSolvers, PoseGraphBackend, ::testing::ValuesIn(availablePoseGraphSolvers()),
  [](const ::testing::TestParamInfo<std::string> & info) {return info.param;});
}  // namespace
