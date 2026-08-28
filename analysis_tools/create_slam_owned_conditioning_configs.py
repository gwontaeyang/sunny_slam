#!/usr/bin/env python3
"""Create fair A/B configs for SLAM-owned LaserScan conditioning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DEFAULT_CANONICAL_CONFIG = (
    REPOSITORY_ROOT
    / "slam_2d_bringup/config/slam_roughness_innovation_params.yaml"
)
SCAN_CONDITIONING_PREFIX = "roughness_scan_conditioning_"
ENABLE_PARAMETER = "enable_roughness_scan_conditioning"
MODE_PARAMETER = "roughness_scan_conditioning_mode"
VARIANTS = {
    "slam_owned_default_off": {
        ENABLE_PARAMETER: False,
        MODE_PARAMETER: "observe",
    },
    "slam_owned_quarantine": {
        ENABLE_PARAMETER: True,
        MODE_PARAMETER: "quarantine",
    },
    "slam_owned_quarantine_recovery3": {
        ENABLE_PARAMETER: True,
        MODE_PARAMETER: "quarantine",
        "roughness_scan_conditioning_quarantine_recovery_scan_count": 3,
    },
    "slam_owned_residual_closer_cap03": {
        ENABLE_PARAMETER: True,
        MODE_PARAMETER: "residual_filter",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parameters(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config is not a mapping: {path}")
    node = payload.get("slam_toolbox")
    if not isinstance(node, dict):
        raise ValueError(f"missing slam_toolbox mapping: {path}")
    parameters = node.get("ros__parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"missing ros__parameters mapping: {path}")
    return payload, parameters


def canonical_conditioning_parameters(path: Path) -> dict[str, Any]:
    _, parameters = load_parameters(path)
    selected = {
        name: value
        for name, value in parameters.items()
        if name == ENABLE_PARAMETER
        or name.startswith(SCAN_CONDITIONING_PREFIX)
    }
    if ENABLE_PARAMETER not in selected or MODE_PARAMETER not in selected:
        raise ValueError(
            f"canonical conditioning parameters incomplete: {path}"
        )
    return selected


def build_variants(
    base_config: Path,
    canonical_config: Path,
) -> dict[str, dict[str, Any]]:
    base_payload, _ = load_parameters(base_config)
    conditioning = canonical_conditioning_parameters(canonical_config)
    outputs: dict[str, dict[str, Any]] = {}
    for name, overrides in VARIANTS.items():
        payload = copy.deepcopy(base_payload)
        parameters = payload["slam_toolbox"]["ros__parameters"]
        parameters.update(conditioning)
        parameters.update(overrides)
        outputs[name] = payload
    return outputs


def write_variants(
    base_config: Path,
    canonical_config: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variants(base_config, canonical_config)
    files: dict[str, dict[str, Any]] = {}
    for name, payload in variants.items():
        path = output_dir / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        files[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "overrides": VARIANTS[name],
        }

    manifest = {
        "schema": 1,
        "base_config": str(base_config.resolve()),
        "base_config_sha256": file_sha256(base_config),
        "canonical_config": str(canonical_config.resolve()),
        "canonical_config_sha256": file_sha256(canonical_config),
        "variant_files": files,
    }
    manifest_path = output_dir / "slam_owned_conditioning_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument(
        "--canonical-config",
        type=Path,
        default=DEFAULT_CANONICAL_CONFIG,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = write_variants(
        args.base_config.expanduser().resolve(),
        args.canonical_config.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
