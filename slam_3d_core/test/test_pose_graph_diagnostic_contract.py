import re
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1]
HEADER_PATH = CORE_ROOT / "include" / "slam_3d_core" / "scan_to_map_icp.hpp"
SOURCE_PATH = CORE_ROOT / "src" / "scan_to_map_icp.cpp"


def test_rejected_place_result_preserves_every_pose_graph_diagnostic():
    header = HEADER_PATH.read_text(encoding="utf-8")
    source = SOURCE_PATH.read_text(encoding="utf-8")

    registration_result_start = header.index("struct RegistrationResult")
    registration_result_end = header.index("struct DeskewStats")
    registration_result = header[registration_result_start:registration_result_end]
    pose_graph_fields = set(
        re.findall(
            r"\b(?:bool|int|double|std::string)\s+(pose_graph_[A-Za-z0-9_]+)\b",
            registration_result,
        )
    )

    restore_start = source.index(
        "const RegistrationResult rejected_place_result = result;"
    )
    restore_end = source.index("attach_input_stats(result);", restore_start)
    restore_block = re.sub(r"\.\s+", ".", source[restore_start:restore_end])
    copied_fields = set(
        re.findall(
            r"result\.(pose_graph_[A-Za-z0-9_]+)\s*=\s*"
            r"rejected_place_result\.\1\s*;",
            restore_block,
        )
    )

    # A rejected candidate must remain unapplied after restoring the accepted
    # frontend pose; every diagnostic explaining that rejection must survive.
    expected_fields = pose_graph_fields - {"pose_graph_correction_applied"}
    assert copied_fields == expected_fields
