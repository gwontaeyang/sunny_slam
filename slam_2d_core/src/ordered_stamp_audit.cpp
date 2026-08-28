#include "slam_2d_core/ordered_stamp_audit.hpp"

#include <algorithm>
#include <cstdint>

namespace slam_2d_core
{

namespace
{

constexpr std::uint64_t kFnv1a64Prime = 1099511628211ULL;

}  // namespace

void OrderedStampAudit::record(const std::int64_t stamp_ns)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_.count == 0U) {
    state_.first_stamp_ns = stamp_ns;
  } else {
    if (stamp_ns == state_.last_stamp_ns) {
      ++state_.duplicate_count;
    } else if (stamp_ns < state_.last_stamp_ns) {
      ++state_.out_of_order_count;
    } else {
      state_.max_positive_gap_ns = std::max(
        state_.max_positive_gap_ns,
        stamp_ns - state_.last_stamp_ns);
    }
  }
  state_.last_stamp_ns = stamp_ns;
  state_.fingerprint_fnv1a64 = updateFingerprint(
    state_.fingerprint_fnv1a64,
    stamp_ns);
  ++state_.count;
}

OrderedStampAuditSnapshot OrderedStampAudit::snapshot() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::uint64_t OrderedStampAudit::updateFingerprint(
  std::uint64_t fingerprint,
  const std::int64_t stamp_ns)
{
  const auto bits = static_cast<std::uint64_t>(stamp_ns);
  for (unsigned int shift = 0; shift < 64U; shift += 8U) {
    fingerprint ^= (bits >> shift) & 0xffU;
    fingerprint *= kFnv1a64Prime;
  }
  return fingerprint;
}

}  // namespace slam_2d_core
