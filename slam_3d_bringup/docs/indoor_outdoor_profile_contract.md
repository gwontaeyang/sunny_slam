# Indoor / Outdoor 3D SLAM Profile Contract

## 원칙

Indoor와 Outdoor SLAM은 별도 알고리즘 fork가 아니다. 두 환경 모두
`slam_3d_core`의 동일한 deskew, registration, keyframe, submap, place
recognition, pose graph 구현을 사용한다. 환경 차이는 bringup profile이
선택하는 센서 구성, topic, motion prior, 공간 scale, registration parameter,
loop acceptance gate로 표현한다.

```text
raw sensors -> ingest -> canonical timed clouds -> shared slam_3d_core
                         ^
                         indoor/outdoor profile selects wiring and parameters
```

공통 코어를 수정할 때는 Indoor와 Outdoor 기본 동작을 모두 유지해야 한다.
환경 이름을 기준으로 알고리즘을 분기하기보다, 필요한 동작을 명시적인
parameter로 추가하고 각 profile에서 값을 선택한다.

## 병렬 개발 및 대화 소유권

여러 Codex 대화나 개발자가 같은 working tree의 `slam_3d_core`를 동시에
수정하면 마지막 write가 앞선 변경을 덮거나, 한 대화가 다른 대화의 중간
상태를 빌드하는 경쟁 조건이 생긴다. Indoor와 Outdoor를 별도 대화에서
진행하더라도 코어 소스 자체를 두 벌로 복사하지 않는다.

- `slam_3d_core`는 한 저장소의 단일 source of truth로 유지한다.
- 같은 working tree에서는 한 시점에 한 담당만 공통 코어를 수정한다.
- 병렬 작업이 필요하면 별도 git worktree와 branch를 만들고, profile 전용
  config, launch, replay artifact를 각자 소유한다.
- 공통 코어 변경은 명시적으로 diff를 검토해 통합한 뒤 shared-core test와
  영향받는 Indoor/Outdoor replay regression을 다시 실행한다.
- Outdoor가 hold인 동안에도 공통 parameter의 기본값과 Outdoor config
  parsing/build contract는 깨지지 않아야 한다. Outdoor field replay는 재개
  시점에 별도 gate로 수행한다.

따라서 이 대화는 Indoor profile과 현재 Indoor 실험을 소유하고, 다른 대화가
Outdoor를 진행한다면 공통 코어 변경 제안은 별도 branch에서 만든 뒤 이쪽
코어 변경과 비교·통합하는 방식으로 운영한다.

## Profile 비교

| 구분 | Indoor | Outdoor |
|---|---|---|
| 대표 환경 | 복도, 문, 벽·바닥·천장, 반복 구조 | 넓은 현장, 경사·요철, 개방 공간, 장거리 loop |
| LiDAR 구성 | XT32 1대 또는 JT128 2대 | JT128 2대 + XT32 1대 |
| IMU | `/imu/data` | `/imu/data` |
| wheel odometry | profile에서 강제로 비활성화 | `/motor_driver_node/odom` 사용 가능 |
| 초기 registration 예측 | bounded LiDAR constant velocity, 선택적 IMU rotation | wheel interpolation + IMU, 필요 시 constant velocity |
| 센서 간 시간차 보정 | dual JT128에서 IMU rotation-only | wheel translation/rotation interpolation + IMU |
| 대표 registration | voxel-surfel, GICP quality A/B | GICP 또는 voxel-surfel |
| 공간 scale | 작은 local map/submap, 좁은 반복 구조 gate | 큰 local map/submap, 장거리 loop gate |
| 주요 위험 | corridor degeneracy, 반복 문·벽, 동적 장애물 | shock, wheel slip, slope, open-space degeneracy |
| 핵심 동기화 | LiDAR–IMU | LiDAR 3대–IMU–wheel–PTP |

## 현재 profile

### `indoor_xt32_single`

- center XT32를 단일 primary cloud로 사용한다.
- `/imu/data`의 검증된 IMU frame과 축 보정을 사용한다.
- wheel subscription과 wheel/IMU EKF를 만들지 않는다.
- bounded LiDAR constant-velocity prediction을 사용할 수 있다.
- 기본 core parameter는
  `slam_3d_core/config/indoor_xt32_single_params.yaml`이다.

Stage 21 Gangnam bag02·bag03 map/localization 회귀는 이 계열의
`XT32 + CV7 IMU + wheel-disabled` 입력 계약이다.

### `indoor_dual_jt128`

- front JT128을 primary, rear JT128을 auxiliary로 사용한다.
- 두 scan start 사이의 motion은 IMU rotation으로만 보상한다.
- wheel translation은 사용하지 않는다.
- 기본 core parameter는
  `slam_3d_core/config/indoor_dual_jt128_params.yaml`이다.

### `custom` outdoor

- front/rear JT128과 center XT32를 함께 사용할 수 있다.
- XT32 기준 timestamp로 auxiliary cloud를 정렬한다.
- wheel odometry bracket/interpolation과 IMU를 motion prior 및 센서 간 보정에
  사용할 수 있다.
- 큰 submap과 장거리 loop를 위한 설정은
  `config/large_scale_outdoor_pipeline.yaml`에서 관리한다.
- live production 사용 전 PTP 상태, 각 센서 extrinsic, IMU clock, wheel slip
  동작을 현장 bag으로 별도 검증해야 한다.

## 설정 소유권

공통 `slam_3d_core`가 소유하는 항목:

- point-time deskew;
- scan-to-local-map registration;
- SE(3) pose와 keyframe;
- overlapping submap;
- place recognition 및 loop geometric verification;
- robust pose graph correction;
- shock quarantine와 terrain evidence.

profile이 소유하는 항목:

- 센서 topology와 raw/canonical topic;
- IMU, wheel odometry 사용 여부와 frame;
- registration backend 및 threshold;
- local-map, keyframe, submap scale;
- loop candidate/acceptance threshold;
- replay completeness와 품질 gate.

## 선택 위치

`hesai_jt128_xt32_3d_slam.launch.py`의
`learning_slam_sensor_profile`이 profile을 선택한다.
`pipeline_config_file=auto`이면 Indoor profile은 `indoor_pipeline.yaml`,
`custom`은 `large_scale_outdoor_pipeline.yaml`을 보고한다. 실제 node parameter
값은 선택된 `learning_slam_params_file`에 의해 결정되므로, 실행 증거에는
profile 이름뿐 아니라 parameter file 경로와 SHA-256도 함께 보존한다.
