#include "slam_3d_core/registration_policy.hpp"

#include "gtest/gtest.h"

namespace
{

TEST(RegistrationPolicy, ValidatesRelocalizationEvenWhenLocalRegistrationWasRejected)
{
  EXPECT_TRUE(
    gole::slam_3d_core::shouldValidateAcceptedPlaceRecognition(
      true,
      true,
      true));
}

TEST(RegistrationPolicy, DoesNotValidateWithoutAnAcceptedPlaceResult)
{
  EXPECT_FALSE(
    gole::slam_3d_core::shouldValidateAcceptedPlaceRecognition(
      false,
      true,
      true));
}

TEST(RegistrationPolicy, RestoresPrePlaceResultWhenBackendRejectsLoop)
{
  EXPECT_TRUE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(true, false, true, false));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(true, false, true, true));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(true, false, false, false));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(false, false, true, false));
}

TEST(RegistrationPolicy, RestoresShockCandidateBeforeGraphWithoutRequiredValidation)
{
  EXPECT_TRUE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(true, true, false, false));
}

TEST(RegistrationPolicy, DoesNotMutatePoseGraphForShockQuarantinedPlaceCandidate)
{
  EXPECT_FALSE(
    gole::slam_3d_core::shouldAttemptPoseGraphCorrection(true, true));
  EXPECT_TRUE(
    gole::slam_3d_core::shouldAttemptPoseGraphCorrection(true, false));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldAttemptPoseGraphCorrection(false, false));
}

TEST(RegistrationPolicy, PlacePoseMetricsUseBoundedCurrentPrediction)
{
  Eigen::Isometry3d prediction = Eigen::Isometry3d::Identity();
  prediction.translation() = Eigen::Vector3d(25.0, -4.0, 1.0);

  Eigen::Isometry3d bounded_place_pose = prediction;
  bounded_place_pose.translation() += Eigen::Vector3d(0.4, -0.2, 0.1);
  bounded_place_pose.linear() =
    Eigen::AngleAxisd(
    5.0 * 3.14159265358979323846 / 180.0,
    Eigen::Vector3d::UnitZ()) * prediction.linear();

  const auto metrics = gole::slam_3d_core::computePlaceRecognitionPoseMetrics(
    prediction, bounded_place_pose);

  EXPECT_TRUE(metrics.finite);
  EXPECT_NEAR(metrics.prediction_jump_translation_m, std::sqrt(0.21), 1.0e-9);
  EXPECT_NEAR(metrics.prediction_jump_rotation_deg, 5.0, 1.0e-9);
}

TEST(RegistrationPolicy, PlacePoseMetricsExposeGlobalMapCorrection)
{
  Eigen::Isometry3d prediction = Eigen::Isometry3d::Identity();
  prediction.translation() = Eigen::Vector3d(30.0, 0.0, 0.0);

  Eigen::Isometry3d bounded_place_pose = prediction;
  bounded_place_pose.linear() =
    Eigen::AngleAxisd(
    10.0 * 3.14159265358979323846 / 180.0,
    Eigen::Vector3d::UnitZ()) * prediction.linear();

  const auto metrics = gole::slam_3d_core::computePlaceRecognitionPoseMetrics(
    prediction, bounded_place_pose);

  EXPECT_TRUE(metrics.finite);
  EXPECT_NEAR(metrics.prediction_jump_translation_m, 0.0, 1.0e-9);
  EXPECT_GT(metrics.map_correction_translation_m, 5.0);
  EXPECT_NEAR(metrics.map_correction_rotation_deg, 10.0, 1.0e-9);
  EXPECT_TRUE(
    gole::slam_3d_core::placePredictionJumpWithinLimits(metrics, 1.2, 20.0));
}

TEST(RegistrationPolicy, PlacePredictionJumpRejectsLargeBoundedRotation)
{
  Eigen::Isometry3d prediction = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d bounded_place_pose = Eigen::Isometry3d::Identity();
  bounded_place_pose.linear() =
    Eigen::AngleAxisd(
    65.0 * 3.14159265358979323846 / 180.0,
    Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto metrics = gole::slam_3d_core::computePlaceRecognitionPoseMetrics(
    prediction, bounded_place_pose);

  EXPECT_FALSE(
    gole::slam_3d_core::placePredictionJumpWithinLimits(metrics, 8.0, 60.0));
}

}  // namespace
