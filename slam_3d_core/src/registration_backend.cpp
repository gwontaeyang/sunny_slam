#include "slam_3d_core/registration_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include <pcl/common/point_tests.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

#ifdef SLAM_3D_HAS_FAST_GICP
#include <fast_gicp/gicp/fast_gicp.hpp>
#endif

namespace gole
{
namespace slam_3d_core
{
namespace
{
using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;

template<typename Base>
class WithIterations : public Base
{
public:
  int iterations() const {return this->nr_iterations_;}
};

#ifdef SLAM_3D_HAS_FAST_GICP
class FastGicpRegistration : public fast_gicp::FastGICP<Point, Point>
{
public:
  static constexpr int covariance_neighbors = 20;

  int iterations() const
  {
    // LsqRegistration stores the zero-based loop index, not the executed count.
    return std::min(this->nr_iterations_ + 1, this->max_iterations_);
  }
};
#endif

class NdtRegistration : public WithIterations<pcl::NormalDistributionsTransform<Point, Point>>
{
public:
  bool hasTargetDistributions()
  {
    const auto & leaves = this->target_cells_.getLeaves();
    return std::any_of(
      leaves.begin(), leaves.end(), [](const auto & item) {
        return item.second.nr_points >= 6;
      });
  }

  bool hasSourceSupport(const Cloud & source, const Eigen::Matrix4f & transform) const
  {
    std::vector<TargetGridLeafConstPtr> neighbors;
    std::vector<float> distances;
    int supported_points = 0;
    for (const auto & point : source) {
      Point transformed;
      transformed.getVector3fMap() = transform.block<3, 3>(0, 0) * point.getVector3fMap() +
        transform.block<3, 1>(0, 3);
      this->target_cells_.radiusSearch(transformed, this->resolution_, neighbors, distances);
      const bool supported = std::any_of(
        neighbors.begin(), neighbors.end(), [](const auto * leaf) {
          return leaf->nr_points >= 6;
        });
      if (supported && ++supported_points >= 3) {
        return true;
      }
    }
    return false;
  }
};

bool positiveFinite(const double value)
{
  return std::isfinite(value) && value > 0.0;
}

template<typename Matcher>
RegistrationBackendResult runMatcher(
  Matcher & matcher, const Cloud::ConstPtr & source, const Cloud::ConstPtr & target,
  const Eigen::Matrix4f & guess, const RegistrationBackendOptions & options)
{
  matcher.setInputSource(source);
  matcher.setInputTarget(target);
  matcher.setMaximumIterations(options.max_iterations);
  matcher.setMaxCorrespondenceDistance(options.max_correspondence_m);
  RegistrationBackendResult result;
  result.target_T_source = guess;
  if constexpr (std::is_same_v<Matcher, NdtRegistration>) {
    // An empty NDT grid can otherwise report a zero-gradient "convergence".
    if (!matcher.hasTargetDistributions() || !matcher.hasSourceSupport(*source, guess)) {
      return result;
    }
  }
  matcher.align(result.aligned, guess);
  result.target_T_source = matcher.getFinalTransformation();
  result.iterations = matcher.iterations();
  result.converged = matcher.hasConverged() && result.target_T_source.allFinite();
  if constexpr (std::is_same_v<Matcher, NdtRegistration>) {
    result.converged = result.converged &&
      matcher.hasSourceSupport(*source, result.target_T_source);
  }
  if (result.target_T_source.allFinite()) {
    // Preserve the existing PCL nearest-neighbor fitness convention for every backend.
    // NDT likelihood and GICP optimization cost are not comparable quality scores.
    result.fitness = matcher.getFitnessScore(options.quality_max_correspondence_m);
  }
  return result;
}
}  // namespace

std::vector<std::string> availableRegistrationMethods()
{
  std::vector<std::string> result{"icp", "gicp", "ndt", "voxel_surfel"};
#ifdef SLAM_3D_HAS_FAST_GICP
  result.emplace_back("fast_gicp");
#endif
  return result;
}

void validateRegistrationMethod(const std::string & method)
{
  const auto available = availableRegistrationMethods();
  if (std::find(available.begin(), available.end(), method) != available.end()) {
    return;
  }
  if (method == "fast_gicp") {
    throw std::invalid_argument(
            "registration_method=fast_gicp is unavailable; install fast_gicp and rebuild "
            "with -DSLAM_3D_WITH_FAST_GICP=ON. No fallback is performed.");
  }
  throw std::invalid_argument(
          "Unknown registration_method='" + method +
          "'; expected icp, gicp, fast_gicp, ndt, or voxel_surfel");
}

void validateRegistrationOptions(
  const std::string & method, const RegistrationBackendOptions & options)
{
  validateRegistrationMethod(method);
  if (options.max_iterations < 1 || !positiveFinite(options.max_correspondence_m) ||
    !positiveFinite(options.quality_max_correspondence_m))
  {
    throw std::invalid_argument("Registration iterations and correspondence limits must be positive");
  }
  if (method == "icp" && options.use_trimmed &&
    (!positiveFinite(options.trim_ratio) || options.trim_ratio > 1.0 ||
    options.min_correspondences < 3))
  {
    throw std::invalid_argument("Invalid ICP trimmed correspondence settings");
  }
  if (method == "fast_gicp" &&
    (options.fast_gicp_num_threads < 1 || options.fast_gicp_num_threads > 256))
  {
    throw std::invalid_argument("fast_gicp_num_threads must be in [1, 256]; 0/all-CPU is disabled");
  }
  if (method == "ndt" &&
    (!positiveFinite(options.ndt_resolution_m) || !positiveFinite(options.ndt_step_size) ||
    !positiveFinite(options.ndt_transformation_epsilon)))
  {
    throw std::invalid_argument(
            "NDT resolution, step size, and transformation epsilon must be finite and positive");
  }
}

RegistrationBackendResult alignRegistration(
  const std::string & method, const Cloud::ConstPtr & source, const Cloud::ConstPtr & target,
  const Eigen::Matrix4f & guess, const RegistrationBackendOptions & options)
{
  validateRegistrationOptions(method, options);
  if (method == "voxel_surfel") {
    throw std::invalid_argument("voxel_surfel uses VoxelSurfelRegistration, not the PCL adapter");
  }
  RegistrationBackendResult rejected;
  rejected.target_T_source = guess;
  const auto finite_cloud = [](const Cloud::ConstPtr & cloud) {
      return cloud && cloud->size() >= 3 &&
             std::all_of(
        cloud->begin(), cloud->end(), [](const Point & point) {
          return pcl::isFinite(point);
        });
    };
  if (!guess.allFinite() || !finite_cloud(source) || !finite_cloud(target)) {
    return rejected;
  }
  if (method == "icp") {
    WithIterations<pcl::IterativeClosestPoint<Point, Point>> matcher;
    if (options.use_trimmed) {
      auto rejector = std::make_shared<pcl::registration::CorrespondenceRejectorTrimmed>();
      rejector->setOverlapRatio(static_cast<float>(options.trim_ratio));
      rejector->setMinCorrespondences(static_cast<unsigned int>(options.min_correspondences));
      matcher.addCorrespondenceRejector(rejector);
    }
    return runMatcher(matcher, source, target, guess, options);
  }
  if (method == "gicp") {
    WithIterations<pcl::GeneralizedIterativeClosestPoint<Point, Point>> matcher;
    return runMatcher(matcher, source, target, guess, options);
  }
#ifdef SLAM_3D_HAS_FAST_GICP
  if (method == "fast_gicp") {
    // FastGICP allocates K covariance neighbors even if the search returns fewer.
    if (source->size() < FastGicpRegistration::covariance_neighbors ||
      target->size() < FastGicpRegistration::covariance_neighbors)
    {
      return rejected;
    }
    FastGicpRegistration matcher;
    matcher.setCorrespondenceRandomness(FastGicpRegistration::covariance_neighbors);
    matcher.setNumThreads(options.fast_gicp_num_threads);
    return runMatcher(matcher, source, target, guess, options);
  }
#endif
  NdtRegistration matcher;
  matcher.setResolution(static_cast<float>(options.ndt_resolution_m));
  matcher.setStepSize(options.ndt_step_size);
  matcher.setTransformationEpsilon(options.ndt_transformation_epsilon);
  return runMatcher(matcher, source, target, guess, options);
}

}  // namespace slam_3d_core
}  // namespace gole
