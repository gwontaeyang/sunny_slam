# slam_3d_map_projection

Offline conversion from an optimized 3D SLAM map to reviewable Nav2 map-server
artifacts. The full PCD remains the localization authority; this package only
creates derived planning products. It does not claim that every merged-PCD
endpoint is a confirmed physical collision.

## Data contract

Inputs:

- finite XYZ PCD in the `map` frame; the cleaned-map workflow uses
  `slam_3d_map_cleanup`'s `map_static.pcd`, not a capture of the temporally
  unfiltered core `/slam_3d/map/global_cloud`;
- post-optimization trajectory CSV with `x_m`, `y_m`, and `z_m` columns, plus
  either `yaw_deg` or `qx`, `qy`, `qz`, `qw` when optimized orientation exists;
  `stamp_ns` and `frame_id` are consumed when present;
- explicit projection YAML.
- optionally, a binary `/slam_3d/terrain/support_evidence` PCD retaining all
  custom scalar fields from `slam_3d_core`.

Outputs:

- `<name>.pgm` and `<name>.yaml`: Nav2 trinary static map in the existing PGM
  contract;
- `<name>.png` and `<name>_png.yaml`: pixel-identical grayscale PNG map and its
  Nav2 metadata. Load `<name>_png.yaml` when the PNG is the desired image;
- `<name>_ramp_mask.pgm` and `<name>_ramp_mask.yaml`: ramp annotation, not a
  keepout map;
- `<name>_terrain_layers.npz`: occupancy, elevation, slope, roughness, point
  counts, accepted and rejected obstacle evidence, terrain-hazard masks, swept
  trajectory footprint, the yaw actually used, segment validity, map origin,
  and resolution;
- `<name>_classified_rgb.pcd`: floor, body obstacle, overhang, below-floor, and
  unreferenced point labels encoded as RGB;
- `<name>_planning_obstacles_rgb.pcd`: only body-height points that contribute
  to final occupied cells. Floor and overhang points are excluded; dynamic
  cleanup must already have been performed by the offline `slam_3d_map_cleanup`
  package. Projection does not verify cleanup provenance or detect people;
- `<name>_metrics.json` and `<name>_review.png`: numeric and visual review evidence.
- with support evidence, `<name>_support_layers.npz`,
  `<name>_traversability_cost.pgm`, `<name>_support_metrics.json`, and
  `<name>_support_review.png`: lossless continuous support/cost layers and a
  four-panel state, vote-ratio, geometry-score, and cost review.

When a timestamped raw bag is available, `raw_scans_to_nav_grid` additionally writes:

- `<name>_visibility_rgb.pcd`: the complete 3D map colored by floor, overhang,
  confirmed obstacle, uncertain obstacle, and miss-contradicted obstacle;
- `<name>.pgm` and `<name>.yaml`: the raw-evidence-refined Nav2 trinary map;
- `<name>_visibility_layers.npz`: per-candidate distinct-scan hit, miss, occluded,
  ambiguous, and unobserved counters plus grid/path provenance;
- `<name>_planned_path.csv`: an offline footprint-aware motion-lattice path used
  to audit map connectivity, not a replacement for Nav2's online global planner;
- `<name>_3d_grid_path_review.png`: paired 3D, top/front, grid, trajectory, and
  generated-path views.

The projection treats both flat floor and detected ramps as `FREE`. Ramp cells are
reported separately so navigation can apply a speed or controller profile without
introducing z as a planning state. High overhangs are retained in the classified PCD
but excluded from the 2D obstacle grid. Unknown cells are never inferred as free from
the absence of PCD points.

The indoor profile keeps `fill_enclosed_unknown_as_free` disabled by default.
Therefore, floor that was not observed remains UNKNOWN even when it is enclosed by
the known floor/wall envelope. The option can be enabled for a reviewed single-floor
map; inferred cells are then retained as `enclosed_unknown_filled_mask` in the
terrain NPZ and highlighted in the review figure. This is topology inference, not
direct ray or floor-return evidence.

## Usage

For temporal-static cleanup, first configure `keyframe_archive_directory` on
the core before mapping, export through private `~/export_keyframes` (default
`/scan_to_map_icp_node/export_keyframes`, `std_srvs/srv/Trigger`), then run
`slam_3d_map_cleanup temporal_map_cleanup` using
[`config/indoor.yaml`](../slam_3d_map_cleanup/config/indoor.yaml).
See the [core migration workflow](../slam_3d_core/README.md#offline-map-cleanup-migration).
The archive reports `snapshot_scope: retained_keyframes` and `pruned_keyframes`;
cleanup cannot recover discarded keyframes. Use `map_static.pcd` and a trajectory
from the same corrected pose state. Cleanup's `map_unfiltered.pcd` contains all
candidate voxel centroids, not an exact copy of the core global cloud.

```bash
ros2 run slam_3d_map_projection pcd_to_nav_grid \
  --pcd /path/to/cleaned_map/map_static.pcd \
  --trajectory /path/to/trajectory_post_optimization.csv \
  --config install/slam_3d_map_projection/share/slam_3d_map_projection/config/indoor_grid_projection.yaml \
  --output-dir /path/to/derived_map \
  --map-name indoor_floor_0 \
  --dataset-label "indoor bag02"
```

To also verify that the resulting free space supports a collision-checked
rectangular-footprint route:

```bash
ros2 run slam_3d_map_projection pcd_to_nav_grid \
  --pcd /path/to/cleaned_map/map_static.pcd \
  --trajectory /path/to/trajectory_post_optimization.csv \
  --config install/slam_3d_map_projection/share/slam_3d_map_projection/config/indoor_grid_projection.yaml \
  --output-dir /path/to/derived_map \
  --map-name indoor_floor_0 \
  --dataset-label "indoor offline-cleaned map" \
  --plan-path
```

`--plan-path` adds `<name>_planned_path.csv`, `<name>_path_layers.npz`, and
`<name>_path_review.png`. It plans between the farthest trajectory poses in the
largest connected footprint-valid component. This is an offline connectivity
test; goal-directed operation still belongs to Nav2. Both `UNKNOWN` and
`OCCUPIED` cells are blocked, while ramp cells remain ordinary `FREE` cells.

For a 2 cm Nav2 raster without repeating all dense floor-layer operations at
2 cm, rerasterize the final obstacle PCD over the validated 5 cm terrain layers:

```bash
ros2 run slam_3d_map_projection refine_nav_grid_resolution \
  --terrain-layers /path/to/map_terrain_layers.npz \
  --planning-obstacles-pcd /path/to/map_planning_obstacles_rgb.pcd \
  --config install/slam_3d_map_projection/share/slam_3d_map_projection/config/indoor_grid_projection.yaml \
  --output-resolution-m 0.02 \
  --obstacle-closing-radius-m 0.10 \
  --enclosure-closing-radius-m 0.16 \
  --no-fill-enclosed-unknown-as-free \
  --output-dir /path/to/nav2_map_2cm \
  --map-name indoor_floor_0_2cm \
  --dataset-label "indoor floor/person filtered"
```

This keeps floor, ramp, hazard, and UNKNOWN semantics from the coarse evidence
grid, but rerasterizes retained static obstacle points at 2 cm. It is not
nearest-neighbor enlargement of the obstacle image. With the conservative flag
shown above, enclosed unobserved floor remains UNKNOWN, including any coarse
topology-completed cells whose provenance mask is available. Fine rasterization
cannot recover point detail already removed by an upstream map voxel filter.
The larger enclosure-closing radius is used only to decide which UNKNOWN regions
are interior. Its additional bridges are recorded in the NPZ and are not exported
wholesale as final obstacles. A non-hazard bridge cell that remains UNKNOWN inside
the completed enclosure is promoted to an inferred wall; exterior bridge cells and
terrain hazards remain UNKNOWN. Final wall geometry otherwise uses the smaller
obstacle-closing radius.

To retain probable ground without converting it to binary free space:

```bash
ros2 run slam_3d_map_projection pcd_to_nav_grid \
  --pcd /path/to/cleaned_map/map_static.pcd \
  --trajectory /path/to/trajectory_post_optimization.csv \
  --config install/slam_3d_map_projection/share/slam_3d_map_projection/config/indoor_grid_projection.yaml \
  --support-evidence-pcd /path/to/terrain_support_evidence.pcd \
  --support-evidence-radius-m 0.10 \
  --probable-support-min-cost 0.35 \
  --output-dir /path/to/derived_map \
  --map-name indoor_floor_0 \
  --dataset-label "indoor bag02"
```

The support PCD state values are `0=unresolved`, `1=probable support`,
`2=confirmed support`, `3=probable hazard`, and `4=confirmed hazard`.
`support_probability` is an empirical independent-view vote ratio, not a
calibrated posterior. `geometric_traversability_score` also includes observation
maturity, roughness, and a ramp penalty, but deliberately excludes robot geometry
and dynamics. The resulting `traversal_cost` remains `NaN` for true unknown,
uses `[0,1]` for finite evidence, floors probable-support cost at the configured
minimum, and uses `1` for hazards. The preview PGM encodes finite costs as
`0..252`, hard hazards as `254`, and unknown as `255`; consume the NPZ when exact
floating-point values are required. `support_cells_overridden_by_hard_hazard`
counts cells where 3D support evidence was superseded by the hard 2D obstacle
rule; the support-state review draws those conflicts in black.

Raw XT32 visibility refinement requires the original bag and a timestamped full-SE(3)
optimized trajectory:

```bash
ros2 run slam_3d_map_projection raw_scans_to_nav_grid \
  --bag /path/to/rosbag2 \
  --pcd /path/to/full_map.pcd \
  --trajectory /path/to/trajectory_post_optimization.csv \
  --projection-config install/slam_3d_map_projection/share/slam_3d_map_projection/config/indoor_grid_projection.yaml \
  --raw-config install/slam_3d_map_projection/share/slam_3d_map_projection/config/indoor_raw_visibility.yaml \
  --output-dir /path/to/paired_products \
  --map-name indoor_floor_0 \
  --dataset-label "indoor bag02"
```

The raw trajectory must include strictly increasing `stamp_ns` and `qx`, `qy`, `qz`,
`qw`. Every map candidate is reprojected at the corresponding raw beam timestamp by
linear translation interpolation and quaternion slerp. The configured LiDAR extrinsic
is therefore a calibration input, not a visualization offset.

Raw returns are accumulated into native ring/azimuth range images. A candidate is a
hit only near the first return, a miss only when the first return lies beyond it by a
margin, and occluded when a nearer return blocks it. No-return beams and occluded
beams never create free space. A coarse XY obstacle is removed only when every height
band represented at that XY is contradicted by distinct scans; it becomes `FREE` only
with local floor support and connectivity to existing free space, otherwise it remains
`UNKNOWN`.

The bundled path is deliberately conservative: `UNKNOWN` and `OCCUPIED` are blocked,
the exact rectangular footprint is precomputed for each heading bin, and all generated
poses are checked again with continuous yaw. Metrics report exact and lattice-quantized
trajectory validity separately, so heading discretization is not misreported as a map
collision.

Trajectory z only bootstraps floor candidates near the driven route. Final point
heights are measured against a locally smoothed floor layer whose seed points pass
PCA normal, planarity, and neighborhood checks. Cells without local floor support
remain unknown instead of inheriting the z of a distant trajectory pose.

The optimized yaw, when present, rasterizes the actual rectangular footprint swept
between poses. Occupied endpoints inside that historical footprint are removed from
the final static grid but retained in a separate provenance mask. This is evidence
that the robot volume was traversable at mapping time, not evidence that it remains
clear forever.

Timestamped CSV rows are sorted by unique `stamp_ns` before use and must share one
nonempty `frame_id` when frame metadata exists. Swept free space is never interpolated
across a pose pair exceeding the configured time, XY, z, or yaw jump. Each endpoint
footprint is still retained as historical evidence. This prevents dropped or
reordered trajectory sections from carving a straight free corridor through walls.

Sparse obstacle returns require neighboring endpoint support. Binary closing may
bridge only a configured raster-scale gap; endpoint, closing-only, dilation-only,
and trajectory-cleared cells remain separate in the NPZ, metrics, and review figure.
Nav2 owns robot inflation. Ramp labels require both a coherent trajectory-height
change and compatible local floor normals. Supported below-floor returns, excessive
floor slope, and excessive local PCA plane-normal RMS residual remain `UNKNOWN`
outside the historical footprint. Unlike local z variance, this roughness measure
does not classify an otherwise smooth inclined plane as rough solely because it is
sloped. The current thresholds are conservative hypotheses until the robot's
ground clearance, step capability, slope limit, and complete collision envelope are
measured.

The PCD-only command does not replace scan-origin ray tracing. A final merged PCD cannot encode
miss rays, observation count, occlusion, scan timestamp, or whether an empty cell
was observed at all. Its map is therefore an endpoint-derived planning candidate,
not calibrated collision truth. The raw command adds pose-conditioned first-return
evidence but still uses thresholded distinct-scan counts rather than a calibrated
probabilistic inverse sensor model. The projector rejects PCD and
trajectory inputs with no points inside the configured bootstrap radius, which
normally indicates a frame mismatch.

## Point and grid semantics

| Class | Main 2D grid |
|---|---|
| floor | free when connected to the driven region |
| ramp | free plus ramp mask |
| body obstacle | occupied |
| high overhang | excluded from the 2D grid |
| below floor | supported cells become unknown unless historically traversed |
| excessive slope or PCA plane residual | unknown unless historically traversed |
| no evidence | unknown |

Do not use the ramp mask as a Nav2 keepout filter. It is intended for a future speed
filter or controller-mode selector. Inflation remains the responsibility of Nav2's
`InflationLayer`. Before deployment, replace the provisional collision height range
with measurements of the complete robot envelope, minimum ground clearance, maximum
step, and allowable slope.

The output is a single-level 2D map. A trajectory whose z values differ by more than
the configured limit inside one XY bin is rejected instead of aliasing stacked floors.
Because unknown cells carry unresolved or unsafe evidence, Nav2 planners consuming
this map must use `allow_unknown: false`. The workspace's top-level indoor/apartment
`navigation.launch.py` now defaults `planner_allow_unknown` to `false`; direct use of
the generic `nav2.launch.py` preserves the old `true` default and must pass
`planner_allow_unknown:=false` explicitly for this map.

The source-level comparison and staged architecture are documented in
[`docs/obstacle_projection_reference_matrix.md`](docs/obstacle_projection_reference_matrix.md).
