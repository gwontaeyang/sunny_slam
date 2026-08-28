import importlib.util
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).with_name("compare_roughness_place_pose_graphs.py")
SPEC = importlib.util.spec_from_file_location(
    "compare_roughness_place_pose_graphs", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_variant(poses):
    return MODULE.Variant(
        label="candidate",
        root=Path("/tmp/candidate"),
        poses=poses,
        map_image=np.full((10, 10), 205, dtype=np.uint8),
        extent=(-1.0, 9.0, -1.0, 9.0),
    )


def test_pair_metrics_wraps_yaw_at_pi():
    variant = make_variant(
        {
            1: {"stamp_s": 1.0, "x_m": 0.0, "y_m": 0.0,
                "yaw_rad": math.pi - 0.1},
            8: {"stamp_s": 8.0, "x_m": 0.3, "y_m": 0.4,
                "yaw_rad": -math.pi + 0.1},
        }
    )

    result = MODULE.pair_metrics(variant, (1, 8))

    assert math.isclose(result["translation_gap_m"], 0.5)
    assert math.isclose(result["yaw_gap_deg"], math.degrees(0.2))


def test_render_indices_are_bounded_and_keep_final_state():
    indices, stride = MODULE.trajectory_render_indices(378)

    assert stride == 3
    assert len(indices) <= MODULE.MAX_RENDERED_TRAJECTORY_POINTS + 1
    assert indices[0] == 0
    assert indices[-1] == 377


def test_draw_variant_uses_scatter_points_without_trajectory_lines():
    poses = {
        state_id: {
            "stamp_s": float(state_id),
            "x_m": state_id * 0.1,
            "y_m": math.sin(state_id * 0.1),
            "yaw_rad": 0.0,
        }
        for state_id in range(20)
    }
    variant = make_variant(poses)
    figure, axis = plt.subplots()

    MODULE.draw_variant(axis, variant, [(1, 18)], (-1.0, 9.0, -1.0, 9.0))

    assert len(axis.lines) == 0
    assert len(axis.collections) == 3
    trajectory_offsets = axis.collections[0].get_offsets()
    assert len(trajectory_offsets) == len(poses)
    plt.close(figure)


def test_empty_trajectory_render_request_is_rejected():
    try:
        MODULE.trajectory_render_indices(0)
    except ValueError as exc:
        assert str(exc) == "state_count must be positive"
    else:
        raise AssertionError("expected ValueError")
