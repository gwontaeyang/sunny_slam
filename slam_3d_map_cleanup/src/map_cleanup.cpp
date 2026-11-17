#include "slam_3d_map_cleanup/map_cleanup.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>

namespace gole
{
namespace slam_3d_map_cleanup
{
namespace
{
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

const std::set<std::string> kConfigKeys{
  "enabled", "voxel_size_m", "min_hit_observations", "min_observation_span_s",
  "min_viewpoint_distance_m", "ray_clearing_enabled", "ray_endpoint_margin_m",
  "max_ray_length_m", "max_rays_per_observation", "prior_free_enabled",
  "min_prior_free_observations", "max_prior_free_observations",
  "prior_free_max_sensor_vertical_drop_m", "promotion_min_hit_ratio",
  "retention_min_hit_ratio", "min_miss_observations_for_demotion"};

struct Scan
{
  fs::path pcd;
  Eigen::Vector3d origin_base;
};

struct Keyframe
{
  std::uint64_t id;
  std::int64_t stamp_ns;
  Eigen::Isometry3d map_from_base;
  std::vector<Scan> scans;
};

struct Archive
{
  std::string map_frame;
  std::string base_frame;
  MapZFilter z_filter;
  YAML::Node snapshot_metadata;
  std::vector<std::string> warnings;
  std::vector<Keyframe> keyframes;
};

void require(const bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

double elapsedMs(const Clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void validateConfig(const TemporalStaticMapConfig & config)
{
  require(validTemporalStaticMapConfig(config), "invalid temporal static map configuration");
  const double maximum_cells = std::ceil(3.0 * config.max_ray_length_m / config.voxel_size_m);
  require(
    std::isfinite(maximum_cells) && maximum_cells <
    std::nextafter(static_cast<double>(std::numeric_limits<std::size_t>::max()), 0.0),
    "configuration exceeds representable ray traversal size");
}

YAML::Node loadYaml(const fs::path & path)
{
  const auto documents = YAML::LoadAllFromFile(path.string());
  require(documents.size() == 1U, "expected exactly one YAML document: " + path.string());
  return documents.front();
}

void checkKeys(
  const YAML::Node & node, const std::set<std::string> & allowed,
  const std::string & context, const bool require_all = true)
{
  require(node.IsMap(), context + " must be a mapping");
  std::set<std::string> seen;
  for (const auto & entry : node) {
    require(entry.first.IsScalar(), context + " has a non-scalar key");
    const auto key = entry.first.as<std::string>();
    require(allowed.count(key) != 0U, context + ": unknown key " + key);
    require(seen.insert(key).second, context + ": duplicate key " + key);
  }
  if (require_all) {
    require(seen == allowed, context + " is missing required fields");
  }
}

template<typename Integer>
Integer integer(const YAML::Node & node, const std::string & context)
{
  require(node.IsScalar(), context + " must be an integer scalar");
  const auto value = node.Scalar();
  Integer output{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
  require(
    parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(),
    context + " must be a representable decimal integer");
  return output;
}

std::vector<double> finiteVector(
  const YAML::Node & node, const std::size_t size, const std::string & context)
{
  require(node.IsSequence() && node.size() == size, context + " has the wrong vector size");
  std::vector<double> result;
  result.reserve(size);
  for (const auto & entry : node) {
    require(entry.IsScalar(), context + " must contain numeric scalars");
    const double value = entry.as<double>();
    require(std::isfinite(value), context + " contains a nonfinite value");
    result.push_back(value);
  }
  return result;
}

bool within(const fs::path & root, const fs::path & child)
{
  auto current = child.begin();
  for (const auto & component : root) {
    if (current == child.end() || *current != component) {
      return false;
    }
    ++current;
  }
  return true;
}

fs::path archiveFile(const fs::path & root, const fs::path & relative)
{
  require(!relative.empty() && !relative.is_absolute(), "archive file path must be relative");
  const auto resolved = fs::canonical(root / relative);
  require(within(root, resolved), "archive file escapes input root: " + relative.string());
  require(fs::is_regular_file(resolved), "archive file is not a regular file: " + relative.string());
  return resolved;
}

void validateZFilter(const MapZFilter & filter)
{
  require(
    std::isfinite(filter.min_m) && std::isfinite(filter.max_m) && filter.min_m <= filter.max_m,
    "invalid map_z_filter bounds");
}

Archive loadArchive(const fs::path & root, const TemporalStaticMapConfig & config)
{
  const auto node = loadYaml(archiveFile(root, "manifest.yaml"));
  checkKeys(node, {"schema_version", "map_frame", "base_frame", "map_z_filter", "keyframes",
      "snapshot_scope", "pruned_keyframes", "complete_session", "point_frame", "point_source"},
    "manifest", false);
  for (const std::string field : {"schema_version", "map_frame", "base_frame", "map_z_filter",
    "keyframes"})
  {
    require(node[field].IsDefined(), "manifest is missing " + field);
  }
  require(integer<unsigned>(node["schema_version"], "schema_version") == 1U,
    "unsupported archive schema_version");
  Archive archive;
  for (const auto & field : std::map<std::string, std::string>{
    {"snapshot_scope", "retained_keyframes"}, {"point_frame", "base_frame"},
    {"point_source", "deskewed_mapping_only_before_terrain_classification"}})
  {
    if (node[field.first]) {
      require(node[field.first].IsScalar() &&
        node[field.first].as<std::string>() == field.second,
        "unsupported archive " + field.first);
      archive.snapshot_metadata[field.first] = field.second;
    } else {
      archive.snapshot_metadata[field.first] = YAML::Node(YAML::NodeType::Null);
    }
  }
  const bool have_pruned = node["pruned_keyframes"].IsDefined();
  const bool have_complete = node["complete_session"].IsDefined();
  const auto pruned = have_pruned ?
    integer<std::uint64_t>(node["pruned_keyframes"], "pruned_keyframes") : 0U;
  const bool complete = have_complete && node["complete_session"].as<bool>();
  archive.snapshot_metadata["pruned_keyframes"] = have_pruned ?
    YAML::Node(pruned) : YAML::Node(YAML::NodeType::Null);
  archive.snapshot_metadata["complete_session"] = have_complete ?
    YAML::Node(complete) : YAML::Node(YAML::NodeType::Null);
  require(!have_complete || !have_pruned || complete == (pruned == 0U),
    "complete_session contradicts pruned_keyframes");
  if (pruned > 0U) {
    archive.warnings.push_back("archive contains retained keyframes only; " +
      std::to_string(pruned) + " keyframes were pruned; result is not a full-session map");
  } else if (have_complete && !complete) {
    archive.warnings.push_back("archive reports incomplete session; result is not a full-session map");
  } else if (!have_pruned || !have_complete) {
    archive.warnings.push_back("archive completeness is unknown; full-session coverage is not verified");
  }
  for (const std::string field : {"map_frame", "base_frame"}) {
    require(node[field].IsScalar(), field + " must be a string");
    const auto frame = node[field].as<std::string>();
    require(frame.find_first_not_of(" \t\n\r") != std::string::npos &&
      frame != "~" && frame != "null", field + " must be nonempty");
    (field == "map_frame" ? archive.map_frame : archive.base_frame) = frame;
  }
  const auto filter = node["map_z_filter"];
  checkKeys(filter, {"enabled", "min_m", "max_m"}, "map_z_filter");
  archive.z_filter = MapZFilter{
    filter["enabled"].as<bool>(), filter["min_m"].as<double>(), filter["max_m"].as<double>()};
  validateZFilter(archive.z_filter);

  const auto keyframes = node["keyframes"];
  require(keyframes.IsSequence() && keyframes.size() > 0U, "keyframes must be a nonempty sequence");
  require(keyframes.size() <= static_cast<std::size_t>(
      std::numeric_limits<int>::max() - config.max_prior_free_observations),
    "too many keyframes for temporal evidence counters");
  archive.keyframes.reserve(keyframes.size());
  for (const auto & entry : keyframes) {
    checkKeys(entry, {"id", "stamp_ns", "pose", "scans"}, "keyframe");
    Keyframe keyframe;
    keyframe.id = integer<std::uint64_t>(entry["id"], "keyframe.id");
    keyframe.stamp_ns = integer<std::int64_t>(entry["stamp_ns"], "keyframe.stamp_ns");
    if (!archive.keyframes.empty()) {
      require(keyframe.id > archive.keyframes.back().id &&
        keyframe.stamp_ns > archive.keyframes.back().stamp_ns,
        "keyframe id and stamp_ns must both be strictly increasing");
    }
    const auto pose = finiteVector(entry["pose"], 7U, "keyframe.pose");
    Eigen::Quaterniond rotation(pose[6], pose[3], pose[4], pose[5]);
    require(std::isfinite(rotation.norm()) && std::abs(rotation.norm() - 1.0) <= 1.0e-6,
      "keyframe.pose quaternion must have unit norm (tolerance 1e-6)");
    rotation.normalize();
    keyframe.map_from_base = Eigen::Isometry3d::Identity();
    keyframe.map_from_base.linear() = rotation.toRotationMatrix();
    keyframe.map_from_base.translation() = Eigen::Vector3d(pose[0], pose[1], pose[2]);
    const auto scans = entry["scans"];
    require(scans.IsSequence() && scans.size() > 0U, "keyframe.scans must be nonempty");
    std::set<fs::path> seen_paths;
    for (const auto & scan : scans) {
      checkKeys(scan, {"pcd", "origin_base"}, "scan");
      require(scan["pcd"].IsScalar(), "scan.pcd must be a relative path string");
      const auto path = archiveFile(root, scan["pcd"].as<std::string>());
      require(seen_paths.insert(path).second, "duplicate PCD view within a keyframe");
      const auto origin = finiteVector(scan["origin_base"], 3U, "scan.origin_base");
      keyframe.scans.push_back(Scan{path, Eigen::Vector3d(origin[0], origin[1], origin[2])});
    }
    archive.keyframes.push_back(std::move(keyframe));
  }
  return archive;
}

template<typename Scalar>
double readCoordinate(const std::uint8_t * data, const bool big_endian)
{
  std::array<std::uint8_t, sizeof(Scalar)> bytes{};
  std::copy_n(data, bytes.size(), bytes.data());
  const std::uint16_t endian_test = 1;
  const bool host_big_endian = *reinterpret_cast<const std::uint8_t *>(&endian_test) == 0;
  if (host_big_endian != big_endian) {
    std::reverse(bytes.begin(), bytes.end());
  }
  Scalar value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return static_cast<double>(value);
}

TemporalStaticMapSensorView readScan(const Scan & scan)
{
  pcl::PCLPointCloud2 cloud;
  require(pcl::io::loadPCDFile(scan.pcd.string(), cloud) == 0,
    "cannot read PCD: " + scan.pcd.string());
  const auto count = static_cast<std::uint64_t>(cloud.width) * cloud.height;
  require(count > 0U && count <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
    "PCD view must contain a representable nonzero point count: " + scan.pcd.string());
  require(cloud.point_step > 0U &&
    static_cast<std::uint64_t>(cloud.row_step) >=
    static_cast<std::uint64_t>(cloud.width) * cloud.point_step &&
    cloud.data.size() == static_cast<std::uint64_t>(cloud.row_step) * cloud.height,
    "inconsistent PCD data size: " + scan.pcd.string());
  std::array<pcl::PCLPointField, 3> fields;
  for (std::size_t axis = 0; axis < fields.size(); ++axis) {
    const std::string name = std::array<std::string, 3>{"x", "y", "z"}[axis];
    std::size_t matches = 0U;
    for (const auto & field : cloud.fields) {
      if (field.name == name) {
        fields[axis] = field;
        ++matches;
      }
    }
    const auto & field = fields[axis];
    require(matches == 1U && field.count == 1U &&
      (field.datatype == pcl::PCLPointField::FLOAT32 ||
      field.datatype == pcl::PCLPointField::FLOAT64),
      "PCD requires scalar floating-point x, y, z fields: " + scan.pcd.string());
    const std::uint64_t size = field.datatype == pcl::PCLPointField::FLOAT32 ? 4U : 8U;
    require(field.offset + size <= cloud.point_step, "PCD coordinate field exceeds point step");
  }
  TemporalStaticMapSensorView view;
  view.sensor_origin = scan.origin_base;
  view.endpoints.reserve(count);
  for (std::uint32_t row = 0U; row < cloud.height; ++row) {
    for (std::uint32_t column = 0U; column < cloud.width; ++column) {
      const auto * data = cloud.data.data() + static_cast<std::size_t>(row) * cloud.row_step +
        static_cast<std::size_t>(column) * cloud.point_step;
      Eigen::Vector3d point;
      for (std::size_t axis = 0U; axis < fields.size(); ++axis) {
        const auto & field = fields[axis];
        point[axis] = field.datatype == pcl::PCLPointField::FLOAT32 ?
          readCoordinate<float>(data + field.offset, cloud.is_bigendian) :
          readCoordinate<double>(data + field.offset, cloud.is_bigendian);
      }
      require(point.allFinite(), "PCD contains a nonfinite XYZ point: " + scan.pcd.string());
      view.endpoints.push_back(point);
    }
  }
  return view;
}

void validatePosition(const Eigen::Vector3d & point, const double voxel_size_m)
{
  require(point.allFinite(), "nonfinite map-frame coordinate");
  const double limit = std::nextafter(
    static_cast<double>(std::numeric_limits<std::int64_t>::max()), 0.0);
  for (int axis = 0; axis < 3; ++axis) {
    require(std::isfinite(point[axis] / voxel_size_m) &&
      std::abs(point[axis] / voxel_size_m) < limit &&
      std::abs(point[axis]) <= static_cast<double>(std::numeric_limits<float>::max()),
      "map-frame coordinate exceeds voxel or output representation");
  }
}

YAML::Node configYaml(const TemporalStaticMapConfig & config)
{
  YAML::Node node(YAML::NodeType::Map);
#define CONFIG_FIELD(name) node[#name] = config.name
  CONFIG_FIELD(enabled);
  CONFIG_FIELD(voxel_size_m);
  CONFIG_FIELD(min_hit_observations);
  CONFIG_FIELD(min_observation_span_s);
  CONFIG_FIELD(min_viewpoint_distance_m);
  CONFIG_FIELD(ray_clearing_enabled);
  CONFIG_FIELD(ray_endpoint_margin_m);
  CONFIG_FIELD(max_ray_length_m);
  CONFIG_FIELD(max_rays_per_observation);
  CONFIG_FIELD(prior_free_enabled);
  CONFIG_FIELD(min_prior_free_observations);
  CONFIG_FIELD(max_prior_free_observations);
  CONFIG_FIELD(prior_free_max_sensor_vertical_drop_m);
  CONFIG_FIELD(promotion_min_hit_ratio);
  CONFIG_FIELD(retention_min_hit_ratio);
  CONFIG_FIELD(min_miss_observations_for_demotion);
#undef CONFIG_FIELD
  return node;
}

YAML::Node statsYaml(const CleanupResult & result, const TemporalStaticMapConfig & config)
{
  YAML::Node node;
  node["schema_version"] = 1;
  node["config"] = configYaml(config);
  node["keyframes"] = result.keyframes;
  node["scans"] = result.scans;
  node["warnings"] = result.warnings;
  node["point_counts"]["archive"] = result.archive_points;
  node["point_counts"]["z_filtered"] = result.z_filtered_points;
  node["point_counts"]["after_z_filter"] = result.archive_points - result.z_filtered_points;
  node["point_counts"]["map_static"] = result.evidence.confirmed_voxels;
  node["point_counts"]["map_unfiltered"] = result.evidence.candidate_voxels;
  node["point_counts"]["map_rejected"] = result.evidence.rejected_voxels;
  node["point_counts"]["map_evidence_rgb"] = result.evidence.candidate_voxels;
  node["timing_ms"]["processing"] = result.processing_ms;
  node["timing_ms"]["pcd_output"] = result.output_ms;
  node["timing_ms"]["total_before_metadata"] = result.processing_ms + result.output_ms;
  const auto & stats = result.evidence;
#define STATS_FIELD(name) node["evidence"][#name] = stats.name
  STATS_FIELD(observations);
  STATS_FIELD(input_points);
  STATS_FIELD(candidate_voxels);
  STATS_FIELD(confirmed_voxels);
  STATS_FIELD(rejected_voxels);
  STATS_FIELD(contradicted_voxels);
  STATS_FIELD(endpoint_hit_updates);
  STATS_FIELD(ray_voxel_checks);
  STATS_FIELD(ray_miss_updates);
  STATS_FIELD(prior_free_voxel_count);
  STATS_FIELD(prior_free_update_count);
  STATS_FIELD(prior_free_seeded_candidate_count);
  STATS_FIELD(promotions);
  STATS_FIELD(demotions);
  STATS_FIELD(last_update_runtime_ms);
  STATS_FIELD(cumulative_update_runtime_ms);
#undef STATS_FIELD
  return node;
}

void writeYaml(const fs::path & path, const YAML::Node & node)
{
  YAML::Emitter emitter;
  emitter.SetDoublePrecision(std::numeric_limits<double>::max_digits10);
  emitter << node;
  require(emitter.good(), "cannot serialize YAML: " + path.string());
  std::ofstream stream;
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream.open(path);
  stream << emitter.c_str() << '\n';
  stream.close();
}

template<typename Point>
void writePcd(const fs::path & path, const pcl::PointCloud<Point> & cloud)
{
  if (cloud.empty()) {
    // PCL binary writers differ in their handling of zero points. Emit a complete
    // PCD header explicitly; never substitute a dummy point for empty evidence.
    std::ofstream stream;
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.open(path, std::ios::binary);
    stream.imbue(std::locale::classic());
    stream << pcl::PCDWriter::generateHeader(cloud) << "DATA ascii\n";
    stream.close();
  } else {
    require(pcl::io::savePCDFileBinary(path.string(), cloud) == 0,
      "cannot write PCD: " + path.string());
  }
}

void writeMaps(const fs::path & output, const TemporalStaticMapAccumulator & accumulator)
{
  pcl::PointCloud<pcl::PointXYZ> all;
  pcl::PointCloud<pcl::PointXYZ> confirmed;
  pcl::PointCloud<pcl::PointXYZ> rejected;
  pcl::PointCloud<pcl::PointXYZRGB> rgb;
  for (const auto & evidence : accumulator.points(TemporalStaticMapSelection::kAll)) {
    const pcl::PointXYZ point(
      static_cast<float>(evidence.position.x()), static_cast<float>(evidence.position.y()),
      static_cast<float>(evidence.position.z()));
    all.push_back(point);
    (evidence.confirmed ? confirmed : rejected).push_back(point);
    pcl::PointXYZRGB color;
    color.x = point.x;
    color.y = point.y;
    color.z = point.z;
    color.r = evidence.confirmed ? 0U : 255U;
    color.g = evidence.confirmed ? 255U : 165U;
    color.b = 0U;
    rgb.push_back(color);
  }
  for (auto * cloud : {&all, &confirmed, &rejected}) {
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1U;
    cloud->is_dense = true;
  }
  rgb.width = static_cast<std::uint32_t>(rgb.size());
  rgb.height = 1U;
  rgb.is_dense = true;
  writePcd(output / "map_static.pcd", confirmed);
  writePcd(output / "map_unfiltered.pcd", all);
  writePcd(output / "map_rejected.pcd", rejected);
  writePcd(output / "map_evidence_rgb.pcd", rgb);
}

}  // namespace

TemporalStaticMapConfig loadCleanupConfig(const std::filesystem::path & yaml_path)
{
  TemporalStaticMapConfig config;
  const auto node = loadYaml(yaml_path);
  checkKeys(node, kConfigKeys, "config", false);
#define READ_CONFIG(name, type) \
  if (node[#name]) {config.name = node[#name].as<type>();}
  READ_CONFIG(enabled, bool);
  READ_CONFIG(voxel_size_m, double);
  READ_CONFIG(min_observation_span_s, double);
  READ_CONFIG(min_viewpoint_distance_m, double);
  READ_CONFIG(ray_clearing_enabled, bool);
  READ_CONFIG(ray_endpoint_margin_m, double);
  READ_CONFIG(max_ray_length_m, double);
  READ_CONFIG(prior_free_enabled, bool);
  READ_CONFIG(prior_free_max_sensor_vertical_drop_m, double);
  READ_CONFIG(promotion_min_hit_ratio, double);
  READ_CONFIG(retention_min_hit_ratio, double);
#undef READ_CONFIG
#define READ_INTEGER(name) \
  if (node[#name]) {config.name = integer<int>(node[#name], #name);}
  READ_INTEGER(min_hit_observations);
  READ_INTEGER(max_rays_per_observation);
  READ_INTEGER(min_prior_free_observations);
  READ_INTEGER(max_prior_free_observations);
  READ_INTEGER(min_miss_observations_for_demotion);
#undef READ_INTEGER
  validateConfig(config);
  return config;
}

TemporalStaticMapSensorView projectSensorView(
  const Eigen::Isometry3d & map_from_base, const TemporalStaticMapSensorView & base_view,
  const MapZFilter & z_filter, const double voxel_size_m)
{
  require(map_from_base.matrix().allFinite() &&
    (map_from_base.linear().transpose() * map_from_base.linear()).isApprox(
      Eigen::Matrix3d::Identity(), 1.0e-6) &&
    std::abs(map_from_base.linear().determinant() - 1.0) <= 1.0e-6 &&
    map_from_base.matrix().row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)),
    "map_from_base must be a finite rigid transform");
  require(std::isfinite(voxel_size_m) && voxel_size_m > 0.0, "invalid voxel_size_m");
  validateZFilter(z_filter);
  require(base_view.sensor_origin.allFinite(), "nonfinite base-frame sensor origin");
  require(!base_view.endpoints.empty(), "empty base-frame sensor view");
  TemporalStaticMapSensorView projected;
  projected.sensor_origin = map_from_base * base_view.sensor_origin;
  validatePosition(projected.sensor_origin, voxel_size_m);
  projected.endpoints.reserve(base_view.endpoints.size());
  for (const auto & endpoint : base_view.endpoints) {
    require(endpoint.allFinite(), "nonfinite base-frame endpoint");
    const Eigen::Vector3d world = map_from_base * endpoint;
    validatePosition(world, voxel_size_m);
    if (!z_filter.enabled || (world.z() >= z_filter.min_m && world.z() <= z_filter.max_m)) {
      projected.endpoints.push_back(world);
    }
  }
  return projected;
}

CleanupResult cleanupArchive(
  const std::filesystem::path & input_dir, const std::filesystem::path & output_dir,
  const TemporalStaticMapConfig & config)
{
  const auto started = Clock::now();
  validateConfig(config);
  const auto input = fs::canonical(input_dir);
  require(fs::is_directory(input), "input must be an archive directory");
  require(!output_dir.empty(), "output-dir must be nonempty");
  const auto requested_output = fs::absolute(output_dir);
  require(!fs::exists(fs::symlink_status(requested_output)), "output path already exists");
  auto output = fs::weakly_canonical(requested_output);
  if (!output.has_filename() && output != output.root_path()) {
    output = output.parent_path();
  }
  require(!within(input, output) && !within(output, input),
    "input and output must not alias or contain one another");
  require(fs::is_directory(output.parent_path()), "output parent directory must already exist");
  const auto archive = loadArchive(input, config);
  TemporalStaticMapAccumulator accumulator(config);
  CleanupResult result;
  result.keyframes = archive.keyframes.size();
  result.warnings = archive.warnings;
  for (const auto & keyframe : archive.keyframes) {
    std::vector<TemporalStaticMapSensorView> views;
    views.reserve(keyframe.scans.size());
    std::size_t observation_points = 0U;
    for (const auto & scan : keyframe.scans) {
      const auto base_view = readScan(scan);
      ++result.scans;
      result.archive_points += base_view.endpoints.size();
      observation_points += base_view.endpoints.size();
      require(observation_points <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "too many points in merged keyframe");
      auto projected = projectSensorView(
        keyframe.map_from_base, base_view, archive.z_filter, config.voxel_size_m);
      result.z_filtered_points += base_view.endpoints.size() - projected.endpoints.size();
      views.push_back(std::move(projected));
    }
    // Relative seconds preserve nanosecond intervals without overflowing int64 subtraction.
    const auto delta_ns = static_cast<long double>(keyframe.stamp_ns) -
      static_cast<long double>(archive.keyframes.front().stamp_ns);
    accumulator.addObservation(static_cast<double>(delta_ns * 1.0e-9L), views);
  }
  result.evidence = accumulator.stats();
  require(result.evidence.candidate_voxels <= std::numeric_limits<std::uint32_t>::max(),
    "too many candidate voxels for PCD width");
  result.processing_ms = elapsedMs(started);

  require(fs::create_directory(output), "output path already exists");
  try {
    const auto writing = Clock::now();
    writeMaps(output, accumulator);
    result.output_ms = elapsedMs(writing);
    auto stats = statsYaml(result, config);
    stats["map_frame"] = archive.map_frame;
    stats["base_frame"] = archive.base_frame;
    stats["output_point_frame"] = archive.map_frame;
    stats["map_z_filter"]["enabled"] = archive.z_filter.enabled;
    stats["map_z_filter"]["min_m"] = archive.z_filter.min_m;
    stats["map_z_filter"]["max_m"] = archive.z_filter.max_m;
    for (const auto & field : archive.snapshot_metadata) {
      stats[field.first.as<std::string>()] = field.second;
    }
    writeYaml(output / "stats.yaml", stats);
    YAML::Node manifest;
    manifest["schema_version"] = 1;
    manifest["status"] = "complete";
    manifest["input_archive"] = input.string();
    manifest["map_frame"] = archive.map_frame;
    manifest["base_frame"] = archive.base_frame;
    manifest["output_point_frame"] = archive.map_frame;
    for (const auto & field : archive.snapshot_metadata) {
      manifest[field.first.as<std::string>()] = field.second;
    }
    manifest["warnings"] = result.warnings;
    manifest["outputs"]["confirmed"] = "map_static.pcd";
    manifest["outputs"]["all_candidates"] = "map_unfiltered.pcd";
    manifest["outputs"]["unconfirmed"] = "map_rejected.pcd";
    manifest["outputs"]["evidence_rgb"] = "map_evidence_rgb.pcd";
    manifest["outputs"]["stats"] = "stats.yaml";
    manifest["unfiltered_semantics"] = "all temporal candidate voxel centroids; not core raw voxelmap";
    manifest["rejected_semantics"] = "unconfirmed temporal evidence; not semantic person-only";
    writeYaml(output / "manifest.yaml.tmp", manifest);
    fs::rename(output / "manifest.yaml.tmp", output / "manifest.yaml");
  } catch (const std::exception & error) {
    try {
      YAML::Node failure;
      failure["status"] = "failed";
      failure["error"] = error.what();
      writeYaml(output / "failure.yaml", failure);
    } catch (...) {
      // Disk or permission failures can also prevent writing the failure marker.
    }
    throw;
  }
  return result;
}

int runTemporalMapCleanupCli(int argc, char ** argv, std::ostream & out, std::ostream & err)
{
  const std::string usage =
    "Usage: temporal_map_cleanup --input ARCHIVE_DIR --output-dir NEW_DIR [--config YAML]\n";
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    out << usage;
    return 0;
  }
  std::map<std::string, fs::path> options;
  for (int index = 1; index < argc; index += 2) {
    const std::string key(argv[index]);
    if ((key != "--input" && key != "--output-dir" && key != "--config") ||
      index + 1 >= argc || std::string(argv[index + 1]).empty() ||
      std::string(argv[index + 1]).rfind("--", 0U) == 0U || options.count(key) != 0U)
    {
      err << "invalid or duplicate argument: " << key << '\n' << usage;
      return 2;
    }
    options.emplace(key, argv[index + 1]);
  }
  if (options.count("--input") == 0U || options.count("--output-dir") == 0U) {
    err << usage;
    return 2;
  }
  const auto config = options.count("--config") != 0U ?
    loadCleanupConfig(options.at("--config")) : TemporalStaticMapConfig{};
  const auto result = cleanupArchive(options.at("--input"), options.at("--output-dir"), config);
  for (const auto & warning : result.warnings) {
    err << "warning: " << warning << '\n';
  }
  out << "complete: " << result.keyframes << " keyframes, " << result.evidence.confirmed_voxels <<
    " confirmed, " << result.evidence.rejected_voxels << " unconfirmed candidate voxels\n" <<
    "output: " << fs::absolute(options.at("--output-dir")).string() << '\n';
  return 0;
}

}  // namespace slam_3d_map_cleanup
}  // namespace gole
