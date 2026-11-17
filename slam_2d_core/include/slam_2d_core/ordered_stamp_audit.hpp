#ifndef SLAM_2D_CORE__ORDERED_STAMP_AUDIT_HPP_
#define SLAM_2D_CORE__ORDERED_STAMP_AUDIT_HPP_

#include <cstdint>
#include <mutex>

namespace slam_2d_core
{

struct OrderedStampAuditSnapshot
{
  std::uint64_t count{0};
  std::int64_t first_stamp_ns{0};
  std::int64_t last_stamp_ns{0};
  std::int64_t max_positive_gap_ns{0};
  std::uint64_t duplicate_count{0};
  std::uint64_t out_of_order_count{0};
  std::uint64_t fingerprint_fnv1a64{14695981039346656037ULL};
};

class OrderedStampAudit
{
public:
  void record(std::int64_t stamp_ns);
  OrderedStampAuditSnapshot snapshot() const;

private:
  static std::uint64_t updateFingerprint(
    std::uint64_t fingerprint,
    std::int64_t stamp_ns);

  mutable std::mutex mutex_;
  OrderedStampAuditSnapshot state_;
};

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__ORDERED_STAMP_AUDIT_HPP_
