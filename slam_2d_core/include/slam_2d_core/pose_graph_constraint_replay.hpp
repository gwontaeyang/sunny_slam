#ifndef SLAM_2D_CORE__POSE_GRAPH_CONSTRAINT_REPLAY_HPP_
#define SLAM_2D_CORE__POSE_GRAPH_CONSTRAINT_REPLAY_HPP_

#include <array>
#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace slam_2d_core
{

struct PoseGraphReplayConstraint
{
  std::string group_label;
  std::string edge_label;
  std::int32_t source_state_id{0};
  std::int32_t target_state_id{0};
  double relative_x_m{0.0};
  double relative_y_m{0.0};
  double relative_yaw_rad{0.0};
  // Row-major symmetric covariance: xx, xy, xyaw, yy, yyaw, yawyaw.
  std::array<double, 6> covariance{};
};

struct PoseGraphReplayParseResult
{
  std::vector<PoseGraphReplayConstraint> constraints;
  std::vector<std::string> errors;
};

PoseGraphReplayParseResult parsePoseGraphReplayConstraints(std::istream & input);

bool isPositiveDefiniteCovariance(const std::array<double, 6> & covariance);

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__POSE_GRAPH_CONSTRAINT_REPLAY_HPP_
