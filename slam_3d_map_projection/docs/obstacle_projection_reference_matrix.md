# Obstacle Projection Reference Matrix

## Decision boundary

The current input is a final merged XYZ PCD plus an optimized trajectory. This is
enough to estimate local floor geometry and classify point endpoints by height. It
is not enough to recover scan origins, miss rays, per-scan visibility, observation
frequency, or occlusion. The current output must therefore expose evidence instead
of presenting every black cell as measured collision truth.

The implementation uses seven obstacle accounting layers:

1. `obstacle_endpoint_mask`: locally floor-referenced, body-height merged-PCD
   endpoints that pass point-count and neighborhood support.
2. `obstacle_bridge_mask`: cells introduced only by binary closing.
3. `obstacle_dilation_mask`: cells introduced only by projector dilation. The
   indoor default is zero because Nav2 owns footprint inflation.
4. `obstacle_cleared_by_trajectory_mask`: endpoint or derived cells removed because
   the optimized rectangular robot footprint physically swept through them.
5. `obstacle_rejected_low_count_mask`: return cells rejected before neighborhood
   support because too few points landed in the cell.
6. `obstacle_rejected_low_support_mask`: count-qualified cells rejected by the local
   endpoint-support test.
7. `obstacle_rejected_far_from_free_mask`: supported endpoints outside the configured
   planning-region distance.

Supported below-floor returns, excessive floor slope, and excessive local PCA
plane-normal RMS residual are retained as separate terrain-hazard masks. The latter
is `sqrt(lambda_min)` of the local 3D covariance and therefore does not conflate a
smooth inclined plane with rough terrain. Outside the historical driven footprint
these hazards remain `UNKNOWN`, not `FREE` or positive occupied evidence. The
thresholds are provisional until measured robot mobility limits are available.

Unknown remains unknown. Small connected components are not deleted by area alone:
a small component can be a table leg, post, door frame, or other real collision.

## Source-level comparison

| Reference | Verified implementation mechanism | Applicable now | Requires raw scans |
|---|---|---:|---:|
| [GroundGrid paper](https://arxiv.org/abs/2405.15664), [source](https://github.com/dcmlr/groundgrid/blob/main/src/GroundSegmentation.cpp) | `detect_ground_patch<S>()` switches between 3x3 and 5x5 patches, gates updates using point count and range-scaled variance, prevents high-confidence ground from moving upward abruptly, fuses height with confidence, and decays confidence during interpolation. Point labeling is relative to the resulting local ground height with distance/variance tolerance. | Yes: local floor layer, support radius, PCA/roughness evidence. A persistent confidence state is not recoverable from one merged PCD. | Per-frame confidence update and visibility-aware outlier rejection. |
| [TRAVEL paper](https://arxiv.org/abs/2206.03190), [core source](https://github.com/url-kaist/TRAVEL/blob/main/cpp/travel/core/travel/tgs.hpp) | Each XY square is split into four tri-grid nodes. Nodes fit an SVD plane. `LocalConvecityConcavity()` checks normal similarity and bidirectional point-to-plane distance; BFS retains traversable graph connectivity. `segmentNodeGround()` measures obstacle height relative to the accepted local plane. | Yes: local plane/normal checks and trajectory-connected floor are compatible. The current regular grid is simpler than the tri-grid graph. | Not inherently, although per-scan application is its intended mode. |
| [Patchwork++ paper](https://arxiv.org/abs/2207.11919), [implementation](https://github.com/url-kaist/patchwork-plusplus/blob/master/cpp/patchworkpp/src/patchworkpp.cpp) | Concentric zones apply adaptive elevation and flatness tests. A-GLE adapts thresholds; TGR reverts likely ground; R-VPF suppresses vertical structures; RNR handles reflected noise. | Possible future ground-seed plugin. It should replace or vote on floor seeds, not directly define static occupied cells. | Best applied before scan merging so one bad scan does not become a permanent endpoint. |
| [Elevation Mapping CuPy paper](https://arxiv.org/abs/2204.12876), [repository](https://github.com/leggedrobotics/elevation_mapping_cupy) | GPU elevation fusion includes height-drift compensation, raycasting visibility cleanup, exclusion zones, overhang artifact handling, and postprocessing plugins. | Local elevation, slope, roughness, and overhang separation are conceptually applicable. | Visibility cleanup, drift compensation over time, and raycasting need timestamped observations and sensor poses. |
| [Traversability Estimation](https://github.com/leggedrobotics/traversability_estimation) | Filter chain derives surface normals, slope, roughness, and step layers from elevation. `check_footprint_path` evaluates a robot footprint over those layers. | Yes after robot-specific maximum step, slope, clearance, and footprint are measured. Current ramp output is annotation only. | No for a static elevation layer, but raw scans improve the layer quality. |
| [MITRE Fast Layered Map](https://github.com/mitre/mitre_fast_layered_map) | `SensorMap` keeps ground/nonground history and occupancy. `BayesUpdate` uses recent-frame hit counts. Its `RayTrace2d` is a map-center visibility heuristic, not a calibrated LiDAR-origin inverse sensor model. The project describes itself as an early/naive implementation. | Evidence separation and the warning against component-size deletion apply now. | Frame history and visibility processing require ordered scans; use an origin-based mapper for collision truth. |
| [OctoMap paper and implementation](https://octomap.github.io/) | Probabilistic octree integrates occupied endpoints and free cells along sensor rays, clamps log odds, and retains unknown space implicitly. | The occupied/free/unknown contract applies. | Correct hit/miss updates require each scan origin. A merged endpoint cloud cannot reconstruct misses. |
| [Wavemap paper](https://arxiv.org/abs/2306.01279), [repository](https://github.com/ethz-asl/wavemap) | Haar-wavelet multi-resolution occupancy supports coarse-to-fine integration and uncertainty-aware measurement models; current integrators target high recall for thin obstacles and multi-sensor fusion. | Useful as a future 3D occupancy backend, not as a post-hoc endpoint filter. | Measurement integration requires calibrated sensor models, origins, and scans. |
| [ROG-Map paper](https://arxiv.org/abs/2302.14819), [repository](https://github.com/hku-mars/ROG-Map) | Sliding multi-resolution occupancy, counter-based map updates, optional raycasting, and incremental obstacle inflation target large high-resolution planning maps. | The separation of mapping from inflation supports leaving inflation to Nav2. | Probabilistic/counter updates and raycasting require registered scans and poses. |
| [D-Map paper](https://doi.org/10.1109/TRO.2023.3323936), [repository](https://github.com/hku-mars/D-Map) | Pose-conditioned depth images distinguish occupied, known-free, and unknown without allocating every voxel. | The known/unknown distinction applies, but a merged PCD cannot recreate the depth images. | Pose and organized per-scan depth observations are required. |
| [Spatio-Temporal Voxel Layer](https://github.com/SteveMacenski/spatio_temporal_voxel_layer), [paper](https://doi.org/10.1177/1729881420910530) | Timestamped OpenVDB voxels decay linearly, exponentially, or persistently; sensor frusta accelerate clearing. This is temporal local obstacle memory, not a static ray-occupancy map. | Keep it as the live Nav2 obstacle layer, separate from the static map. | Timestamped observations, sensor frusta, and poses are required for decay/clearing. |
| [Potential Negative Obstacle Detection by Occlusion Labeling](https://publications.ri.cmu.edu/potential-negative-obstacle-detection-by-occlusion-labeling/) | Accumulates 3D data, propagates occluders, and labels missing data using context instead of treating endpoint absence as a drop-off. | Only the conservative rule that below-floor or missing evidence must not be called free. | Negative obstacle evidence fundamentally needs line-of-sight and occlusion information. |
| [CMU Ground Based Autonomy](https://github.com/jizhang-cmu/ground_based_autonomy_basic) | Terrain analysis exposes `noDataObstacle` as a conservative negative-obstacle fallback and warns that registered-scan quality directly controls terrain-map quality. | A separate unknown/possible-drop layer is compatible. Treating every no-data cell as occupied would be too conservative for the current static map. | Reliable drop-off reasoning needs scan geometry or a downward-looking sensor. |
| [Nav2 planner selection guide](https://docs.nav2.org/setup_guides/algorithm/select_algorithm.html), [Smac source documentation](https://github.com/ros-navigation/navigation2/tree/main/nav2_smac_planner) | Smac Hybrid-A* and State Lattice search SE(2) with angle-quantized full-footprint collision checks and kinematically valid motion models. The official guide distinguishes these from point or circular grid planners for non-circular robots. | The offline audit now blocks unknown, uses heading-binned rectangular footprints, checks swept turns, and rechecks the dense continuous-yaw path. | Deployment must use the selected Nav2 Smac model, measured minimum turning radius, actual footprint, inflation costs, and controller constraints. The bundled path is diagnostic only. |

## Implemented architecture

```text
optimized merged PCD + optimized XYZ/yaw trajectory
  -> trajectory-z bootstrap candidates near the driven route
  -> local PCA floor validation
  -> smoothed local floor elevation/support
  -> exclusive point classes relative to local floor
  -> below-floor / excessive-slope / excessive-roughness hazard masks
  -> supported body-height endpoint cells
  -> provenance-preserving closing/dilation
  -> timestamp/order validation and discontinuity-bounded trajectory sections
  -> optimized-yaw swept rectangular footprint clear within valid sections
  -> Nav2 trinary map + ramp/elevation/evidence review layers
```

This is the strongest defensible projection from the available final products. In
particular, a nearest trajectory z is not used as a global floor reference: points
without nearby local floor support are `UNREFERENCED` and project to `UNKNOWN`.
A single-level guard rejects trajectories with incompatible z levels in one XY bin.
The exported terrain NPZ stores the actual yaw used and every segment's interpolation
validity so the cleared corridor is reproducible.

## Implemented raw XT32 visibility refinement

The retained indoor bags make a stronger audit possible. The XT32 PointCloud2 schema
contains `x`, `y`, `z`, `intensity`,
`ring`, and per-point `timestamp`; each scan has 128,000 points in `hesai_center`.
Using the captured `base_link -> hesai_center` transform `(0.36, 0, 1.5496) m` and
`+90 deg` yaw, sampled raw scans align to the optimized merged map with approximately
`0.089 m` median nearest-point distance. The deliberately inverted/no-rotation
extrinsic variants are substantially worse, which validates the transform direction.

The implemented command consumes every configured raw scan and point. For each scan
it builds a nearest-first-return ring/azimuth range image, reprojects merged-map
candidates using full-SE(3) pose interpolation at the beam timestamp, and records one
of `HIT`, `MISS`, `OCCLUDED`, `AMBIGUOUS`, or `UNOBSERVED` per candidate. Counts are
scan-distinct: duplicate points in one packet cannot increase persistence.

Candidates are aggregated in 20 cm XY cells and configurable height bands. A map
endpoint is contradicted only when every represented height band passes the miss-count,
hit-fraction, and miss-to-hit-ratio gates. A nearer raw return is occlusion, not clearing;
an empty range-image bin is unobserved, not clearing. Contradicted cells become free
only where a locally validated floor exists and connects to established free space.
All retained direct endpoints survive morphological closing, and any reconstructed
bridge crossing the optimized historical footprint is cleared again.

The paired offline path audit blocks both unknown and occupied cells. It precomputes
the rectangular footprint at discrete headings, reports heading-lattice and
continuous-yaw validity separately, and verifies every generated path pose with the
continuous footprint. This path demonstrates static-map connectivity only; Nav2 still
owns the deployed planner, inflation, controller, and live STVL obstacles.

## Next architecture

The current refinement is a conservative endpoint audit, not yet a full Bayesian 3D
occupancy backend. The next mapper should move the same evidence contract into online
or replay-time map integration:

```text
deskewed scan + sensor origin + optimized pose
  -> per-scan ground segmentation (local elevation plus Patchwork++/TRAVEL vote)
  -> 3D or height-banded ray integration
  -> per-cell hit_count, miss_count, last_seen, distinct_scan_count, calibrated log_odds
  -> temporal persistence and visibility cleanup
  -> local elevation, slope, plane-residual roughness, step, overhang, drop-off evidence
  -> measured 3D robot-envelope collision projection
  -> static Nav2 layer; STVL remains the live obstacle layer
```

Minimum acceptance rules for that stage:

- Preserve hit, miss, unknown, and occluded states separately.
- Require obstacle persistence from distinct scans, not duplicate points in one
  accumulated map voxel.
- Do not clear behind the first return on a ray.
- Keep negative-obstacle suspicion separate from positive occupied evidence.
- Evaluate collision against a measured height-dependent robot envelope.
- Export raw evidence counters and before/after plots for every threshold decision.
- Replace the current offline diagnostic lattice with the deployed Nav2 planner's
  exact footprint, minimum turning radius, cost model, and map-filter configuration.

## Unverified deployment constants

The current `1.066 m x 0.88 m` footprint follows the active Nav2 configuration that
was inspected during development. The following values are not yet backed by physical
measurement and therefore remain deployment blockers:

- complete robot collision height and overhang envelope;
- minimum ground clearance;
- maximum vertical step and gap;
- maximum safe pitch/roll and sustained ramp slope;
- LiDAR blind zones and self-occlusion for XT32 and both JT128 mounting options.
