#!/usr/bin/env python3
"""Small synthetic archives only; no ROS graph, bag replay, or hardware."""

import copy
import math
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import tempfile
import unittest

import yaml


BINARY = Path(sys.argv.pop(1)).resolve()


def write_pcd(path, points, *, binary=False, double=False):
    size = 8 if double else 4
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\n"
        f"SIZE {size} {size} {size}\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\nDATA {'binary' if binary else 'ascii'}\n"
    )
    if binary:
        data = b"".join(struct.pack("<" + ("ddd" if double else "fff"), *p) for p in points)
    else:
        data = "".join(" ".join(map(str, p)) + "\n" for p in points).encode()
    path.write_bytes(header.encode() + data)


def read_pcd(path):
    with path.open("rb") as stream:
        header = {}
        while True:
            line = stream.readline().decode("ascii").strip()
            if not line or line.startswith("#"):
                if not line:
                    raise AssertionError("missing PCD DATA header")
                continue
            key, *values = line.split()
            header[key] = values
            if key == "DATA":
                break
        count = int(header["POINTS"][0])
        assert int(header["WIDTH"][0]) * int(header["HEIGHT"][0]) == count
        assert header["FIELDS"][:3] == ["x", "y", "z"]
        if header["DATA"] == ["ascii"]:
            records = [list(map(float, line.split())) for line in stream if line.strip()]
            assert len(records) == count
            return header, records
        assert header["DATA"] == ["binary"]
        sizes = list(map(int, header["SIZE"]))
        assert sizes == [4] * len(sizes)
        assert header["COUNT"] == ["1"] * len(sizes)
        data = stream.read()
        stride = sum(sizes)
        assert len(data) == count * stride
        records = []
        for index in range(count):
            point = list(struct.unpack_from("<fff", data, index * stride))
            if "rgb" in header["FIELDS"]:
                rgb = struct.unpack_from("<I", data, index * stride + 12)[0]
                point.append(((rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255))
            records.append(point)
        return header, records


class CleanupCli(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="map_cleanup_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "archive"
        (self.archive / "keyframes").mkdir(parents=True)
        self.output = self.root / "result"
        self.scan = self.archive / "keyframes" / "000000_sensor00.pcd"
        write_pcd(self.scan, [(2.1, 0.1, 0.1)])
        self.document = {
            "schema_version": 1,
            "map_frame": "map",
            "base_frame": "base_link",
            "map_z_filter": {"enabled": False, "min_m": -1.0, "max_m": 2.0},
            "keyframes": [
                {
                    "id": index,
                    "stamp_ns": 1700000000000000000 + index * 1000000000,
                    "pose": [0, 0, 0, 0, 0, 0, 1],
                    "scans": [{"pcd": str(self.scan.relative_to(self.archive)),
                               "origin_base": [0, 0, 0]}],
                }
                for index in range(3)
            ],
        }

    def run_cli(self, config=None, *, output=None, extra=()):
        (self.archive / "manifest.yaml").write_text(yaml.safe_dump(self.document))
        command = [str(BINARY), "--input", str(self.archive), "--output-dir",
                   str(output or self.output)]
        if config is not None:
            path = self.root / "config.yaml"
            path.write_text(yaml.safe_dump(config) if isinstance(config, dict) else config)
            command.extend(["--config", str(path)])
        return subprocess.run(command + list(extra), capture_output=True, text=True, timeout=10)

    def success(self, **kwargs):
        result = self.run_cli(**kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = kwargs.get("output", self.output)
        manifest = yaml.safe_load((output / "manifest.yaml").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse((output / "failure.yaml").exists())
        stats = yaml.safe_load((output / "stats.yaml").read_text())
        for name in ("map_static", "map_unfiltered", "map_rejected", "map_evidence_rgb"):
            _, points = read_pcd(output / (name + ".pcd"))
            self.assertEqual(len(points), stats["point_counts"][name])
        return stats

    def invalid(self, **kwargs):
        result = self.run_cli(**kwargs)
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertFalse(self.output.exists(), result.stderr)
        return result

    def test_default_config_and_output_partition(self):
        write_pcd(self.scan, [(2.1, 0.1, 0.1), (4.1, 0.1, 0.1)], binary=True)
        second = self.archive / "keyframes" / "later.pcd"
        write_pcd(second, [(2.1, 0.1, 0.1)])
        for frame in self.document["keyframes"][1:]:
            frame["scans"][0]["pcd"] = "keyframes/later.pcd"
        stats = self.success()
        self.assertEqual(stats["point_counts"]["map_static"], 1)
        self.assertEqual(stats["point_counts"]["map_rejected"], 1)
        self.assertEqual(stats["point_counts"]["archive"], 4)
        self.assertEqual(stats["evidence"]["observations"], 3)
        self.assertEqual(stats["config"]["voxel_size_m"], 0.22)
        self.assertEqual(stats["config"]["retention_min_hit_ratio"], 0.3)
        self.assertEqual(stats["config"]["prior_free_max_sensor_vertical_drop_m"], 0.9)
        self.assertTrue(stats["config"]["enabled"])
        self.assertTrue(stats["config"]["prior_free_enabled"])
        _, colors = read_pcd(self.output / "map_evidence_rgb.pcd")
        self.assertEqual([p[3] for p in colors], [(0, 255, 0), (255, 165, 0)])
        for value in stats["timing_ms"].values():
            self.assertGreaterEqual(value, 0)

    def test_multisensor_one_vote_and_empty_confirmed_pcd(self):
        second = self.archive / "keyframes" / "000000_sensor01.pcd"
        write_pcd(second, [(2.15, 0.15, 0.15)])
        self.document["keyframes"] = self.document["keyframes"][:1]
        self.document["keyframes"][0]["scans"].append(
            {"pcd": "keyframes/000000_sensor01.pcd", "origin_base": [1, 0, 0]})
        stats = self.success()
        self.assertEqual(stats["evidence"]["observations"], 1)
        self.assertEqual(stats["evidence"]["endpoint_hit_updates"], 1)
        self.assertEqual(stats["point_counts"]["map_static"], 0)
        self.assertEqual(stats["point_counts"]["map_unfiltered"], 1)
        header, points = read_pcd(self.output / "map_static.pcd")
        self.assertEqual(header["WIDTH"], ["0"])
        self.assertEqual(header["HEIGHT"], ["1"])
        self.assertEqual(points, [])

    def test_post_loop_poses_reproject_endpoints(self):
        self.success(config={"ray_clearing_enabled": False})
        _, before = read_pcd(self.output / "map_static.pcd")
        for frame in self.document["keyframes"]:
            frame["pose"] = [10, 20, 3, 0, 0, math.sqrt(0.5), math.sqrt(0.5)]
        optimized = self.root / "optimized"
        self.success(config={"ray_clearing_enabled": False}, output=optimized)
        _, after = read_pcd(optimized / "map_static.pcd")
        self.assertAlmostEqual(before[0][0], 2.1, places=5)
        for actual, expected in zip(after[0], [9.9, 22.1, 3.1]):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_transformed_sensor_origin_drives_ray_evidence(self):
        self.document["keyframes"] = self.document["keyframes"][:2]
        write_pcd(self.scan, [(3.1, 0.1, 0.1)])
        far = self.archive / "keyframes" / "far.pcd"
        write_pcd(far, [(5.1, 0.1, 0.1)])
        self.document["keyframes"][1]["scans"][0]["pcd"] = "keyframes/far.pcd"
        for frame in self.document["keyframes"]:
            frame["pose"] = [10, 20, 0, 0, 0, math.sqrt(0.5), math.sqrt(0.5)]
            frame["scans"][0]["origin_base"] = [1.1, 0.1, 0.1]
        stats = self.success(config={"voxel_size_m": 0.5, "prior_free_enabled": False})
        self.assertEqual(stats["evidence"]["ray_miss_updates"], 1)

    def test_world_z_filter_removes_endpoints_and_their_rays(self):
        self.document["keyframes"] = self.document["keyframes"][:2]
        self.document["map_z_filter"] = {"enabled": True, "min_m": 1, "max_m": 1.5}
        write_pcd(self.scan, [(3.1, 0.1, 0.1)])
        far = self.archive / "keyframes" / "far.pcd"
        write_pcd(far, [(5.1, 0.1, 0.6)])
        self.document["keyframes"][1]["scans"][0]["pcd"] = "keyframes/far.pcd"
        for frame in self.document["keyframes"]:
            frame["pose"][2] = 1
        stats = self.success()
        self.assertEqual(stats["point_counts"]["z_filtered"], 1)
        self.assertEqual(stats["point_counts"]["map_unfiltered"], 1)
        self.assertEqual(stats["evidence"]["ray_miss_updates"], 0)
        _, points = read_pcd(self.output / "map_unfiltered.pcd")
        self.assertAlmostEqual(points[0][2], 1.1, places=5)

    def test_all_z_filtered_outputs_are_valid_empty_pcd(self):
        self.document["map_z_filter"] = {"enabled": True, "min_m": 10, "max_m": 11}
        stats = self.success()
        self.assertEqual(stats["point_counts"]["z_filtered"], 3)
        self.assertEqual(stats["evidence"]["observations"], 3)
        self.assertEqual(stats["point_counts"]["map_evidence_rgb"], 0)

    def test_float64_binary_pcd(self):
        write_pcd(self.scan, [(2.1, 0.1, 0.1)], binary=True, double=True)
        self.assertEqual(self.success()["point_counts"]["map_static"], 1)

    def test_retained_keyframe_metadata_and_pruning_warning(self):
        metadata = {"snapshot_scope": "retained_keyframes", "pruned_keyframes": 317,
                    "complete_session": False, "point_frame": "base_frame",
                    "point_source": "deskewed_mapping_only_before_terrain_classification"}
        self.document.update(metadata)
        for index, frame in enumerate(self.document["keyframes"]):
            frame["id"] = 317 + index
        stats = self.success()
        manifest = yaml.safe_load((self.output / "manifest.yaml").read_text())
        for name, value in metadata.items():
            self.assertEqual(stats[name], value)
            self.assertEqual(manifest[name], value)
        self.assertIn("317", stats["warnings"][0])
        result = self.run_cli(output=self.root / "another")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not a full-session map", result.stderr)

    def test_missing_completeness_is_reported_unknown(self):
        stats = self.success()
        self.assertIsNone(stats["complete_session"])
        self.assertIsNone(stats["pruned_keyframes"])
        self.assertIn("unknown", stats["warnings"][0])

    def test_invalid_or_contradictory_metadata_rejected(self):
        for metadata in ({"snapshot_scope": "whole_map"}, {"pruned_keyframes": -1},
                         {"pruned_keyframes": 2, "complete_session": True},
                         {"pruned_keyframes": 0, "complete_session": False},
                         {"point_frame": "map"}, {"point_source": "raw"},
                         {"complete_session": "maybe"}):
            with self.subTest(metadata=metadata):
                original = copy.deepcopy(self.document)
                self.document.update(metadata)
                self.invalid()
                self.document = original

    def test_stamp_precision_and_noncontiguous_unsigned_ids(self):
        for index, frame in enumerate(self.document["keyframes"]):
            frame["stamp_ns"] = 1700000000000000000 + index
            frame["id"] = (1 << 63) + index * 2
        stats = self.success(config={"min_observation_span_s": 2e-9,
                                     "min_viewpoint_distance_m": 1000})
        self.assertEqual(stats["point_counts"]["map_static"], 1)

    def test_disabled_accumulation_still_validates_archive(self):
        stats = self.success(config={"enabled": False})
        self.assertEqual(stats["evidence"]["observations"], 0)
        self.assertEqual(stats["point_counts"]["archive"], 3)
        self.assertEqual(stats["point_counts"]["map_unfiltered"], 0)

    def test_invalid_manifest_variants_fail_closed(self):
        original = copy.deepcopy(self.document)
        variants = []
        for key, value in (("schema_version", 2), ("schema_version", -1),
                           ("map_frame", ""), ("base_frame", None), ("keyframes", []),
                           ("keyframes", {})):
            doc = copy.deepcopy(original)
            doc[key] = value
            variants.append(doc)
        for key, value in (("pose", [0] * 7), ("pose", [0, 0, 0, 0, 0, 0, 2]),
                           ("pose", [0, 0, math.nan, 0, 0, 0, 1]), ("pose", [0] * 6),
                           ("scans", []), ("id", -1), ("id", 1 << 64),
                           ("stamp_ns", 1 << 63), ("stamp_ns", "1.5")):
            doc = copy.deepcopy(original)
            doc["keyframes"][0][key] = value
            variants.append(doc)
        for key in ("id", "stamp_ns"):
            for delta in (0, -1):
                doc = copy.deepcopy(original)
                doc["keyframes"][1][key] = doc["keyframes"][0][key] + delta
                variants.append(doc)
        for field, value in (("origin_base", [0, math.inf, 0]), ("origin_base", []),
                             ("pcd", "keyframes/missing.pcd"), ("pcd", str(self.scan))):
            doc = copy.deepcopy(original)
            doc["keyframes"][0]["scans"][0][field] = value
            variants.append(doc)
        for key, value in (("min_m", math.nan), ("min_m", 10), ("enabled", "maybe")):
            doc = copy.deepcopy(original)
            doc["map_z_filter"][key] = value
            variants.append(doc)
        for index, doc in enumerate(variants):
            with self.subTest(variant=index):
                self.document = doc
                self.invalid()

    def test_missing_duplicate_unknown_and_multidocument_yaml(self):
        del self.document["keyframes"][0]["scans"][0]["origin_base"]
        self.invalid()
        self.document["keyframes"][0]["scans"][0]["origin_base"] = [0, 0, 0]
        self.document["unknown"] = True
        self.invalid()
        del self.document["unknown"]
        self.run_cli(extra=["--unknown", "x"])
        path = self.archive / "manifest.yaml"
        for suffix in ("schema_version: 1\n", "---\nschema_version: 1\n"):
            path.write_text(yaml.safe_dump(self.document) + suffix)
            result = subprocess.run([str(BINARY), "--input", str(self.archive),
                                     "--output-dir", str(self.output)], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.output.exists())

    def test_out_of_root_and_symlink_pcdfiles_rejected(self):
        outside = self.root / "outside.pcd"
        write_pcd(outside, [(2.1, 0.1, 0.1)])
        self.document["keyframes"][0]["scans"][0]["pcd"] = "../outside.pcd"
        self.invalid()
        link = self.archive / "keyframes" / "escape.pcd"
        link.symlink_to(outside)
        self.document["keyframes"][0]["scans"][0]["pcd"] = "keyframes/escape.pcd"
        self.invalid()

    def test_input_alias_and_output_within_archive_rejected(self):
        link = self.root / "alias"
        link.symlink_to(self.archive, target_is_directory=True)
        for output in (self.archive, link, self.archive / "result", self.root):
            with self.subTest(output=str(output)):
                result = self.run_cli(output=output)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((output / "map_static.pcd").exists())

    def test_existing_output_directory_file_and_dangling_symlink_protected(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("unchanged")
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual(sentinel.read_text(), "unchanged")
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        for name, symlink in (("file", False), ("dangling", True)):
            path = self.root / name
            if symlink:
                path.symlink_to(self.root / "missing")
            else:
                path.write_text("unchanged")
            self.assertNotEqual(self.run_cli(output=path).returncode, 0)
            self.assertTrue(path.is_symlink() if symlink else path.read_text() == "unchanged")

    def test_second_success_cannot_overwrite_first(self):
        self.success()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_trailing_separator_output_directory(self):
        result = self.run_cli(output=str(self.output) + "/")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output / "manifest.yaml").exists())

    def test_output_parent_symlink_resolves_before_dotdot(self):
        (self.archive / "subdir").mkdir()
        link = self.root / "link"
        link.symlink_to(self.archive / "subdir", target_is_directory=True)
        result = self.run_cli(output=link / ".." / "result")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((self.archive / "result").exists())
        self.assertFalse(self.output.exists())

    def test_invalid_config_rejected(self):
        variants = [
            "", "null", "[]", "enabled: true\nenabled: false\n", "{}\n---\n{}\n",
            {"ros__parameters": {}}, {"voxel_size": 0.2}, {"voxel_size_m": 0},
            {"voxel_size_m": math.nan}, {"voxel_size_m": 1e-300}, {"min_hit_observations": 0},
            {"min_hit_observations": 1.5}, {"min_observation_span_s": -1},
            {"min_viewpoint_distance_m": -1}, {"ray_endpoint_margin_m": -1},
            {"max_ray_length_m": 0}, {"max_rays_per_observation": -1},
            {"min_prior_free_observations": 0}, {"max_prior_free_observations": 3},
            {"prior_free_max_sensor_vertical_drop_m": -1}, {"promotion_min_hit_ratio": 0},
            {"retention_min_hit_ratio": 0.9}, {"min_miss_observations_for_demotion": 0},
            {"enabled": "perhaps"},
        ]
        for config in variants:
            with self.subTest(config=config):
                self.invalid(config=config)

    def test_pcd_empty_nonfinite_missing_xyz_and_bad_body_rejected(self):
        for points in ([], [(math.nan, 0, 0)], [(math.inf, 0, 0)]):
            write_pcd(self.scan, points)
            self.invalid()
        write_pcd(self.scan, [(1, 2, 3)])
        self.scan.write_text(self.scan.read_text().replace("FIELDS x y z", "FIELDS x y intensity"))
        self.invalid()
        self.scan.write_text("not a PCD\n")
        self.invalid()

    def test_truncated_pcd_bodies_fail_closed(self):
        for binary in (False, True):
            with self.subTest(binary=binary):
                write_pcd(self.scan, [(1, 2, 3), (4, 5, 6)], binary=binary)
                data = self.scan.read_bytes()
                self.scan.write_bytes(data[:-12] if binary else data.rsplit(b"4 5 6\n", 1)[0])
                self.invalid()

    def test_failed_output_has_no_success_manifest(self):
        (self.archive / "manifest.yaml").write_text(yaml.safe_dump(self.document))

        def limit_output_size():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (128, 128))

        result = subprocess.run(
            [str(BINARY), "--input", str(self.archive), "--output-dir", str(self.output)],
            capture_output=True, text=True, timeout=10, preexec_fn=limit_output_size)
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "manifest.yaml").exists())
        self.assertNotEqual(self.run_cli().returncode, 0)

    def test_help_and_argument_errors(self):
        help_result = subprocess.run([str(BINARY), "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--input", help_result.stdout)
        for args in ([], ["--input"], ["--bad", "x"], ["--help", "extra"],
                     ["--input", "a", "--input", "b", "--output-dir", "c"]):
            result = subprocess.run([str(BINARY)] + args, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
