#ifndef SLAM_2D_CORE__KARTO_SCAN_MATCHER_VECTOR_HPP_
#define SLAM_2D_CORE__KARTO_SCAN_MATCHER_VECTOR_HPP_

#include "slam_toolbox/slam_mapper.hpp"

namespace karto
{

// ROS Humble's karto shared library does not export this specialization.
template<>
kt_double ScanMatcher::MatchScan<LocalizedRangeScanVector>(
  LocalizedRangeScan * pScan,
  const LocalizedRangeScanVector & rBaseScans,
  Pose2 & rMean,
  Matrix3 & rCovariance,
  kt_bool doPenalize,
  kt_bool doRefineMatch);

}  // namespace karto

#endif  // SLAM_2D_CORE__KARTO_SCAN_MATCHER_VECTOR_HPP_
