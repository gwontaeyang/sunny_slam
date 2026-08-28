#include "slam_2d_core/pose_graph_constraint_replay.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace slam_2d_core
{
namespace
{

constexpr const char * kExpectedHeader =
  "group_label,edge_label,source_state_id,target_state_id,relative_x_m,relative_y_m,"
  "relative_yaw_rad,covariance_xx,covariance_xy,covariance_xyaw,covariance_yy,"
  "covariance_yyaw,covariance_yawyaw";

std::string trim(const std::string & value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1U);
}

std::vector<std::string> splitCsvRow(const std::string & row)
{
  std::vector<std::string> fields;
  std::stringstream stream(row);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(trim(field));
  }
  if (!row.empty() && row.back() == ',') {
    fields.emplace_back();
  }
  return fields;
}

bool validLabel(const std::string & label)
{
  if (label.empty()) {
    return false;
  }
  return std::all_of(
    label.begin(), label.end(),
    [](const unsigned char character) {
      return std::isalnum(character) != 0 || character == '_' || character == '-' ||
      character == '.';
    });
}

bool parseInt32(const std::string & text, std::int32_t & value)
{
  if (text.empty()) {
    return false;
  }
  char * end = nullptr;
  errno = 0;
  const long parsed = std::strtol(text.c_str(), &end, 10);
  if (
    errno != 0 || end == text.c_str() || *end != '\0' || parsed < 0 ||
    parsed > std::numeric_limits<std::int32_t>::max())
  {
    return false;
  }
  value = static_cast<std::int32_t>(parsed);
  return true;
}

bool parseFiniteDouble(const std::string & text, double & value)
{
  if (text.empty()) {
    return false;
  }
  char * end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  value = parsed;
  return true;
}

std::string lineError(const std::size_t line_number, const std::string & message)
{
  return "line " + std::to_string(line_number) + ": " + message;
}

}  // namespace

bool isPositiveDefiniteCovariance(const std::array<double, 6> & covariance)
{
  if (!std::all_of(
      covariance.begin(), covariance.end(), [](const double value) {
        return std::isfinite(value);
      }))
  {
    return false;
  }

  const double xx = covariance[0];
  const double xy = covariance[1];
  const double xyaw = covariance[2];
  const double yy = covariance[3];
  const double yyaw = covariance[4];
  const double yawyaw = covariance[5];
  const double leading_minor_2 = xx * yy - xy * xy;
  const double determinant =
    xx * (yy * yawyaw - yyaw * yyaw) -
    xy * (xy * yawyaw - yyaw * xyaw) +
    xyaw * (xy * yyaw - yy * xyaw);
  return xx > 0.0 && leading_minor_2 > 0.0 && determinant > 0.0;
}

PoseGraphReplayParseResult parsePoseGraphReplayConstraints(std::istream & input)
{
  PoseGraphReplayParseResult result;
  std::string row;
  std::size_t line_number = 0U;
  bool header_seen = false;
  std::set<std::pair<std::int32_t, std::int32_t>> unordered_pairs;

  while (std::getline(input, row)) {
    ++line_number;
    const std::string stripped = trim(row);
    if (stripped.empty() || stripped.front() == '#') {
      continue;
    }
    if (!header_seen) {
      header_seen = true;
      if (stripped != kExpectedHeader) {
        result.errors.push_back(lineError(line_number, "unexpected CSV header"));
      }
      continue;
    }

    const std::vector<std::string> fields = splitCsvRow(stripped);
    if (fields.size() != 13U) {
      result.errors.push_back(
        lineError(line_number, "expected 13 fields, got " + std::to_string(fields.size())));
      continue;
    }
    PoseGraphReplayConstraint constraint;
    constraint.group_label = fields[0];
    constraint.edge_label = fields[1];
    bool row_valid = true;
    if (!validLabel(constraint.group_label)) {
      result.errors.push_back(lineError(line_number, "invalid group_label"));
      row_valid = false;
    }
    if (!validLabel(constraint.edge_label)) {
      result.errors.push_back(lineError(line_number, "invalid edge_label"));
      row_valid = false;
    }
    if (!parseInt32(fields[2], constraint.source_state_id)) {
      result.errors.push_back(lineError(line_number, "invalid source_state_id"));
      row_valid = false;
    }
    if (!parseInt32(fields[3], constraint.target_state_id)) {
      result.errors.push_back(lineError(line_number, "invalid target_state_id"));
      row_valid = false;
    }
    double * numeric_fields[] = {
      &constraint.relative_x_m,
      &constraint.relative_y_m,
      &constraint.relative_yaw_rad,
      &constraint.covariance[0],
      &constraint.covariance[1],
      &constraint.covariance[2],
      &constraint.covariance[3],
      &constraint.covariance[4],
      &constraint.covariance[5],
    };
    for (std::size_t index = 0U; index < 9U; ++index) {
      if (!parseFiniteDouble(fields[index + 4U], *numeric_fields[index])) {
        result.errors.push_back(
          lineError(line_number, "invalid numeric field at column " + std::to_string(index + 5U)));
        row_valid = false;
      }
    }
    if (!row_valid) {
      continue;
    }
    if (constraint.source_state_id == constraint.target_state_id) {
      result.errors.push_back(lineError(line_number, "self-edge is not allowed"));
      continue;
    }
    if (!isPositiveDefiniteCovariance(constraint.covariance)) {
      result.errors.push_back(lineError(line_number, "covariance is not positive definite"));
      continue;
    }
    const auto pair = std::minmax(constraint.source_state_id, constraint.target_state_id);
    if (!unordered_pairs.insert(pair).second) {
      result.errors.push_back(lineError(line_number, "duplicate unordered state pair"));
      continue;
    }
    result.constraints.push_back(constraint);
  }

  if (!header_seen) {
    result.errors.emplace_back("missing CSV header");
  } else if (result.constraints.empty() && result.errors.empty()) {
    result.errors.emplace_back("CSV contains no constraints");
  }
  if (!result.errors.empty()) {
    result.constraints.clear();
  }
  return result;
}

}  // namespace slam_2d_core
