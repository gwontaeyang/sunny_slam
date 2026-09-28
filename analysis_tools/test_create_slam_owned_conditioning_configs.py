import importlib.util
import sys
from pathlib import Path

import yaml


SCRIPT = Path(__file__).with_name(
    "create_slam_owned_conditioning_configs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "create_slam_owned_conditioning_configs",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_config(path, parameters):
    path.write_text(
        yaml.safe_dump(
            {"slam_toolbox": {"ros__parameters": parameters}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_variants_change_only_conditioning_parameters(tmp_path):
    base = tmp_path / "base.yaml"
    canonical = tmp_path / "canonical.yaml"
    write_config(
        base,
        {
            "scan_topic": "/scan_laserscan",
            "resolution": 0.02,
            "roughness_score_gate": 2.75,
        },
    )
    write_config(
        canonical,
        {
            "enable_roughness_scan_conditioning": False,
            "roughness_scan_conditioning_mode": "observe",
            "roughness_scan_conditioning_residual_direction": "closer",
            "roughness_scan_conditioning_residual_max_removed_ratio": 0.03,
            "roughness_scan_conditioning_quarantine_recovery_scan_count": 5,
            "roughness_score_gate": 99.0,
        },
    )

    variants = MODULE.build_variants(base, canonical)

    assert set(variants) == set(MODULE.VARIANTS)
    for name, payload in variants.items():
        parameters = payload["slam_toolbox"]["ros__parameters"]
        assert parameters["scan_topic"] == "/scan_laserscan"
        assert parameters["resolution"] == 0.02
        assert parameters["roughness_score_gate"] == 2.75
        assert parameters[
            "roughness_scan_conditioning_residual_direction"
        ] == "closer"
        assert parameters[
            "roughness_scan_conditioning_residual_max_removed_ratio"
        ] == 0.03
        assert parameters[
            "roughness_scan_conditioning_quarantine_recovery_scan_count"
        ] == MODULE.VARIANTS[name].get(
            "roughness_scan_conditioning_quarantine_recovery_scan_count",
            5,
        )
        assert parameters[
            "enable_roughness_scan_conditioning"
        ] == MODULE.VARIANTS[name][
            "enable_roughness_scan_conditioning"
        ]
        assert parameters[
            "roughness_scan_conditioning_mode"
        ] == MODULE.VARIANTS[name]["roughness_scan_conditioning_mode"]


def test_write_variants_records_reproducible_manifest(tmp_path):
    base = tmp_path / "base.yaml"
    canonical = tmp_path / "canonical.yaml"
    output = tmp_path / "output"
    write_config(base, {"scan_topic": "/scan_laserscan"})
    write_config(
        canonical,
        {
            "enable_roughness_scan_conditioning": False,
            "roughness_scan_conditioning_mode": "observe",
        },
    )

    manifest = MODULE.write_variants(base, canonical, output)

    assert manifest["base_config_sha256"] == MODULE.file_sha256(base)
    assert manifest["canonical_config_sha256"] == MODULE.file_sha256(
        canonical
    )
    for name in MODULE.VARIANTS:
        path = output / f"{name}.yaml"
        assert path.is_file()
        assert manifest["variant_files"][name]["sha256"] == (
            MODULE.file_sha256(path)
        )
    assert (output / "slam_owned_conditioning_manifest.json").is_file()
