#ifndef SLAM_2D_CORE__KARTO_LOOP_CLOSURE_OBSERVER_HPP_
#define SLAM_2D_CORE__KARTO_LOOP_CLOSURE_OBSERVER_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "karto_sdk/Mapper.h"

namespace slam_2d_core
{

enum class KartoLoopClosureDisposition
{
  kPending,
  kCoarseRejected,
  kFineRejected,
  kAcceptedIncomplete,
  kAccepted
};

const char * kartoLoopClosureDispositionName(
  KartoLoopClosureDisposition disposition) noexcept;

struct KartoLoopClosureSourceSnapshot
{
  std::int32_t state_id{-1};
  std::int32_t unique_id{-1};
  karto::Pose2 corrected_pose;
  karto::Pose2 odometric_pose;
};

struct KartoLoopClosureAttempt
{
  std::size_t candidate_index{0};
  std::string sensor_name;
  std::vector<std::int32_t> chain_state_ids;
  std::vector<std::int32_t> chain_unique_ids;
  std::vector<KartoLoopClosureSourceSnapshot> source_snapshots;
  karto::Pose2 pre_target_corrected_pose;
  karto::Pose2 pre_target_odometric_pose;
  bool candidate_chain_available{false};
  bool coarse_message_parsed{false};
  double coarse_response{std::numeric_limits<double>::quiet_NaN()};
  double coarse_response_threshold{std::numeric_limits<double>::quiet_NaN()};
  double coarse_covariance_xx_m2{std::numeric_limits<double>::quiet_NaN()};
  double coarse_covariance_yy_m2{std::numeric_limits<double>::quiet_NaN()};
  double coarse_covariance_threshold_m2{
    std::numeric_limits<double>::quiet_NaN()};
  bool coarse_accepted{false};
  bool fine_message_parsed{false};
  double fine_response{std::numeric_limits<double>::quiet_NaN()};
  double fine_response_threshold{std::numeric_limits<double>::quiet_NaN()};
  bool fine_accepted{false};
  bool rejection_callback_seen{false};
  bool begin_callback_seen{false};
  bool end_callback_seen{false};
  bool listener_sequence_complete{false};
  KartoLoopClosureDisposition disposition{KartoLoopClosureDisposition::kPending};

  std::int32_t source_state_id{-1};
  std::int32_t source_unique_id{-1};
  std::int32_t target_state_id{-1};
  std::int32_t target_unique_id{-1};
  bool actual_edge_present{false};
  bool accepted_without_edge{false};
  bool link_info_available{false};
  bool edge_pairing_ambiguous{false};
  bool event_edge_count_before_available{false};
  std::size_t event_edge_count_before{0};
  std::size_t event_edge_count_after{0};
  std::size_t event_new_edge_count{0};
  std::size_t event_edge_index{std::numeric_limits<std::size_t>::max()};
  karto::Pose2 relative_transform;
  karto::Pose2 odometric_relative_transform;
  karto::Pose2 pre_graph_relative_transform;
  karto::Pose2 post_graph_relative_transform;
  karto::Matrix3 factor_covariance;
  double measurement_vs_odometry_translation_m{
    std::numeric_limits<double>::quiet_NaN()};
  double measurement_vs_odometry_yaw_rad{
    std::numeric_limits<double>::quiet_NaN()};
  double measurement_vs_odometry_covariance_weighted_disagreement_sq{
    std::numeric_limits<double>::quiet_NaN()};
  double pre_residual_x_m{std::numeric_limits<double>::quiet_NaN()};
  double pre_residual_y_m{std::numeric_limits<double>::quiet_NaN()};
  double pre_residual_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double post_residual_x_m{std::numeric_limits<double>::quiet_NaN()};
  double post_residual_y_m{std::numeric_limits<double>::quiet_NaN()};
  double post_residual_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double post_covariance_weighted_residual_sq{
    std::numeric_limits<double>::quiet_NaN()};
};

struct KartoLoopClosureObservationBatch
{
  bool observer_requested{false};
  bool listener_attached{false};
  bool listener_detached{false};
  bool available{false};
  bool single_sensor_context{false};
  bool candidate_identity_complete{false};
  bool listener_sequence_complete{false};
  bool edge_pairing_complete{false};
  std::string failure_reason;
  std::int64_t stamp_ns{0};
  std::int32_t target_state_id{-1};
  std::int32_t target_unique_id{-1};
  std::size_t accepted_callback_count{0};
  std::size_t actual_loop_edge_count{0};
  std::size_t accepted_without_edge_count{0};
  std::size_t ambiguous_loop_edge_count{0};
  std::size_t unpaired_loop_edge_count{0};
  std::vector<KartoLoopClosureAttempt> attempts;
};

bool parseKartoLoopClosureCoarseMessage(
  const std::string & message,
  double & response,
  double & response_threshold,
  double & covariance_xx,
  double & covariance_yy,
  double & covariance_threshold) noexcept;

bool parseKartoLoopClosureFineMessage(
  const std::string & message,
  double & response,
  double & response_threshold) noexcept;

class KartoLoopClosureObserver final : public karto::MapperLoopClosureListener
{
public:
  KartoLoopClosureObserver() = default;
  KartoLoopClosureObserver(
    karto::Mapper * mapper,
    karto::LocalizedRangeScan * current_scan,
    std::int64_t stamp_ns);
  ~KartoLoopClosureObserver();

  KartoLoopClosureObserver(const KartoLoopClosureObserver &) = delete;
  KartoLoopClosureObserver & operator=(const KartoLoopClosureObserver &) = delete;

  bool attach(
    karto::Mapper * mapper,
    karto::LocalizedRangeScan * current_scan,
    std::int64_t stamp_ns) noexcept;
  bool detach() noexcept;
  KartoLoopClosureObservationBatch finish() noexcept;

  void LoopClosureCheck(const std::string & info) override;
  void BeginLoopClosure(const std::string & info) override;
  void EndLoopClosure(const std::string & info) override;

private:
  void appendFailure(const std::string & reason) noexcept;
  void finalizeOpenAttempt() noexcept;
  void captureCandidateChain(KartoLoopClosureAttempt & attempt) noexcept;
  void captureAcceptedOutcome(KartoLoopClosureAttempt & attempt) noexcept;

  karto::Mapper * mapper_{nullptr};
  karto::LocalizedRangeScan * current_scan_{nullptr};
  kt_int32u candidate_scan_index_{0};
  bool attached_{false};
  bool finished_{false};
  KartoLoopClosureObservationBatch batch_;
};

void updateKartoLoopClosureOutcomes(
  KartoLoopClosureObservationBatch & batch,
  karto::Mapper * mapper,
  const karto::LocalizedRangeScan * processed_scan) noexcept;

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__KARTO_LOOP_CLOSURE_OBSERVER_HPP_
