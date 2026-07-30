/*
 * This specialization follows OpenKarto's LGPL-3.0 ScanMatcher::MatchScan
 * implementation. ROS 2 Humble declares the template but does not export the
 * LocalizedRangeScanVector specialization from libkartoSlamToolbox.
 */

#include "slam_2d_core/roughness_anchor_slam_toolbox.hpp"

#include <cassert>

namespace karto
{

template<>
kt_double ScanMatcher::MatchScan<LocalizedRangeScanVector>(
  LocalizedRangeScan * pScan,
  const LocalizedRangeScanVector & rBaseScans,
  Pose2 & rMean,
  Matrix3 & rCovariance,
  kt_bool doPenalize,
  kt_bool doRefineMatch)
{
  const Pose2 scan_pose = pScan->GetSensorPose();
  if (pScan->GetNumberOfRangeReadings() == 0) {
    rMean = scan_pose;
    rCovariance.Clear();
    rCovariance(0, 0) = 500.0;
    rCovariance(1, 1) = 500.0;
    rCovariance(2, 2) =
      4.0 * math::Square(m_pMapper->m_pCoarseAngleResolution->GetValue());
    return 0.0;
  }

  const Rectangle2<kt_int32s> roi = m_pCorrelationGrid->GetROI();
  Vector2<kt_double> offset;
  offset.SetX(
    scan_pose.GetX() -
    0.5 * (roi.GetWidth() - 1) * m_pCorrelationGrid->GetResolution());
  offset.SetY(
    scan_pose.GetY() -
    0.5 * (roi.GetHeight() - 1) * m_pCorrelationGrid->GetResolution());
  m_pCorrelationGrid->GetCoordinateConverter()->SetOffset(offset);

  AddScans(rBaseScans, scan_pose.GetPosition());

  const Vector2<kt_double> search_dimensions(
    m_pSearchSpaceProbs->GetWidth(), m_pSearchSpaceProbs->GetHeight());
  const Vector2<kt_double> coarse_search_offset(
    0.5 * (search_dimensions.GetX() - 1) * m_pCorrelationGrid->GetResolution(),
    0.5 * (search_dimensions.GetY() - 1) * m_pCorrelationGrid->GetResolution());
  const Vector2<kt_double> coarse_search_resolution(
    2.0 * m_pCorrelationGrid->GetResolution(),
    2.0 * m_pCorrelationGrid->GetResolution());

  kt_double best_response = CorrelateScan(
    pScan,
    scan_pose,
    coarse_search_offset,
    coarse_search_resolution,
    m_pMapper->m_pCoarseSearchAngleOffset->GetValue(),
    m_pMapper->m_pCoarseAngleResolution->GetValue(),
    doPenalize,
    rMean,
    rCovariance,
    false);

  if (
    m_pMapper->m_pUseResponseExpansion->GetValue() &&
    math::DoubleEqual(best_response, 0.0))
  {
    kt_double search_angle_offset =
      m_pMapper->m_pCoarseSearchAngleOffset->GetValue();
    for (kt_int32u attempt = 0; attempt < 3; ++attempt) {
      search_angle_offset += math::DegreesToRadians(20.0);
      best_response = CorrelateScan(
        pScan,
        scan_pose,
        coarse_search_offset,
        coarse_search_resolution,
        search_angle_offset,
        m_pMapper->m_pCoarseAngleResolution->GetValue(),
        doPenalize,
        rMean,
        rCovariance,
        false);
      if (!math::DoubleEqual(best_response, 0.0)) {
        break;
      }
    }
  }

  if (doRefineMatch) {
    const Vector2<kt_double> fine_search_offset(coarse_search_resolution * 0.5);
    const Vector2<kt_double> fine_search_resolution(
      m_pCorrelationGrid->GetResolution(),
      m_pCorrelationGrid->GetResolution());
    best_response = CorrelateScan(
      pScan,
      rMean,
      fine_search_offset,
      fine_search_resolution,
      0.5 * m_pMapper->m_pCoarseAngleResolution->GetValue(),
      m_pMapper->m_pFineSearchAngleOffset->GetValue(),
      doPenalize,
      rMean,
      rCovariance,
      true);
  }

  assert(math::InRange(rMean.GetHeading(), -KT_PI, KT_PI));
  return best_response;
}

}  // namespace karto
