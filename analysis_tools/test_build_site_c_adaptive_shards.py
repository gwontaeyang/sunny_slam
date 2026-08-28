#!/usr/bin/env python3

import argparse
import unittest

from analysis_tools.build_site_c_adaptive_shards import (
    HISTORICAL_TOPIC_COUNTS,
    SegmentSpec,
    historical_count_check,
    historical_non_scan_timestamp_allowed,
    historical_static_timestamp_allowed,
    parse_segment,
    validate_segments,
)


class SiteCAdaptiveShardTest(unittest.TestCase):
    def test_parse_segment_uses_exclusive_end(self):
        segment = parse_segment("site_c_q3:14133:19867")
        self.assertEqual(segment.name, "site_c_q3")
        self.assertEqual(segment.scan_start, 14133)
        self.assertEqual(segment.scan_end_exclusive, 19867)
        self.assertEqual(segment.scan_end_exclusive - segment.scan_start, 5734)

    def test_invalid_segment_is_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_segment("site_c_q3:12:12")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_segment("nested/name:0:10")

    def test_segments_are_sorted_and_must_not_overlap(self):
        ordered = validate_segments(
            [SegmentSpec("q3", 20, 30), SegmentSpec("q1", 0, 10)]
        )
        self.assertEqual([item.name for item in ordered], ["q1", "q3"])
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_segments(
                [SegmentSpec("q1", 0, 10), SegmentSpec("q2", 9, 20)]
            )

    def test_historical_contract_matches_exact_counts(self):
        counts = dict(HISTORICAL_TOPIC_COUNTS["site_c_q1"])
        result = historical_count_check("site_c_q1", counts)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["mismatches"], {})

    def test_historical_contract_reports_count_mismatch(self):
        counts = dict(HISTORICAL_TOPIC_COUNTS["site_c_q3"])
        counts["/scan_laserscan"] -= 1
        result = historical_count_check("site_c_q3", counts)
        self.assertEqual(result["status"], "mismatched")
        self.assertEqual(
            result["mismatches"]["/scan_laserscan"],
            {"expected": 5734, "actual": 5733},
        )

    def test_custom_segment_has_no_historical_contract(self):
        result = historical_count_check("custom", {"/scan_laserscan": 10})
        self.assertEqual(result["status"], "not_available")

    def test_historical_non_scan_timestamp_uses_frozen_exclusive_end(self):
        start = 1785821483679498517
        end = 1785821819611225179
        self.assertFalse(historical_non_scan_timestamp_allowed("site_c_q1", start - 1))
        self.assertTrue(historical_non_scan_timestamp_allowed("site_c_q1", start))
        self.assertTrue(historical_non_scan_timestamp_allowed("site_c_q1", end - 1))
        self.assertFalse(historical_non_scan_timestamp_allowed("site_c_q1", end))
        self.assertTrue(historical_static_timestamp_allowed("site_c_q1", start - 1))
        self.assertTrue(historical_non_scan_timestamp_allowed("custom", end))


if __name__ == "__main__":
    unittest.main()
