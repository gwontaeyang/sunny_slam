# 현재 2D SLAM 기능

현재 로컬 `slam_2d_core`와 `slam_2d_bringup`에 구현된 기능을 설명한다.

## 기본 SLAM

`sensor_msgs/msg/LaserScan`을 입력으로 2D 점유지도를 생성하고 로봇의 위치와 방향을 추정한다. 기본 실행 파일은 `slam_2d_core/roughness_aware_slam_toolbox_node`이며, ROS 노드 이름은 `slam_toolbox`로 유지한다.

직접 만든 frontend에서 scan 처리와 roughness 관련 로직을 담당한다. 지도 생성, scan matching, pose graph 및 최적화에는 SLAM Toolbox, Karto, Ceres를 사용한다.

- 기본 scan 입력: `/scan_laserscan`
- 지도 좌표계: `map`
- odometry 좌표계: `odom`
- 로봇 기준 좌표계: `base_link`
- 지도 해상도: 기본 `0.02 m`
- 지도 갱신 주기: 기본 `0.5 s`
- LaserScan 사용 거리: 기본 `0.1 ~ 30.0 m`

## Scan Matching과 Pose Graph

연속으로 들어오는 scan을 정합해 로봇의 상대 이동을 계산하고, pose graph에 scan과 pose를 누적한다. 가까운 scan chain 사이의 제약과 재방문 loop-closure 제약을 구성하며, Ceres 최적화 결과를 pose와 지도에 반영한다.

- 연속 scan matching으로 이동 중의 위치·방향을 추정한다.
- Near-chain matching으로 주변 scan chain과의 연결 관계를 구성한다.
- Loop closing으로 이전에 방문한 영역과의 정합 관계를 반영한다.
- 최적화된 pose graph를 바탕으로 점유지도를 갱신한다.
- 기존 scan matching과 loop closing은 기본 활성화 상태다.

## IMU Roughness 인식

`/imu/data`를 scan 시각 주변의 시간창으로 모아 흔들림 정도를 계산한다. 기울기, roll/pitch 방향 각속도, 가속도 크기의 중력 대비 편차와 변화율을 함께 사용한다.

IMU는 기본 mapping 경로에서 pose를 직접 대체하는 입력이 아니다. 흔들림이 발생한 구간을 표시하고, 그 구간의 scan과 제약을 해석하는 보조 신호로 사용한다.

## Adaptive Roughness Memory

흔들림이 발생한 뒤에도 해당 구간의 영향을 연속적인 memory strength로 유지한다. 시간 경과, 이동 거리, roughness 강도와 안정 상태 회복에 따라 memory가 감소한다.

Memory와 scan-match 품질은 별도로 다룬다. Memory가 높다는 이유만으로 정상 제약을 제거하지 않으며, IMU나 후보 품질 정보가 없으면 기존 제약을 유지하는 fail-open 동작을 한다.

Constraint 처리 모드는 `intervention_off`, `legacy_fixed_memory`, `adaptive_memory_observe`, `adaptive_memory_only`, `adaptive_quality_aware`로 구분돼 있다. 기본 live launch는 `adaptive_memory_observe`를 사용하며, 관찰 결과만으로 graph를 변경하지 않는다.

## 실제 Constraint 후보 관찰

Karto 내부에서 만들어지는 실제 near-chain 후보와 loop-closure 후보를 관찰한다. Sequential preview와 실제 graph 연결 후보를 구분한다.

- Source/target state ID와 후보 chain 종류
- Scan-match response와 상대 변환
- Covariance와 odometry/prediction 대비 변환 차이
- Memory strength와 유효 시간·거리
- 후보 처리 결과와 graph 연결 관계
- 최적화 전후 residual 정보

반복되는 near-chain의 회전 불일치도 추적한다. 품질 정보와 roughness prior를 함께 사용하는 경로와 graph를 바꾸지 않는 관찰 경로가 분리돼 있다.

## 흔들림 전후 Scan Chain 기반 장소 인식

흔들린 scan 자체를 같은 장소의 확정 증거로 사용하지 않는다. 흔들림 전후의 안정적인 scan chain을 구성하고, 이전 구간과 같은 passage인지 확인하는 reference-chain observer가 구현돼 있다.

안정 구간의 정·역방향 scan matching, response, covariance, 상대 변환 차이와 통과 전후 이동의 일관성을 함께 확인한다. 같은 passage에서 연속으로 나온 중복 근거는 독립적인 장소 인식 근거와 구분한다.

Reference-chain observer는 선택 기능이며 기본 비활성화다.

## SLAM 내부 Scan Conditioning

Scan conditioning은 다른 perception 또는 localization 노드가 아니라 `slam_2d_core` 안에 구현돼 있다. Roughness 상태와 scan 품질에 따라 관찰, scan 억제·간격 조절, 안정 scan 기반 처리, residual filtering과 quarantine/recovery 동작을 선택할 수 있다.

기본값은 비활성화다. 기존 LaserScan 입력은 그대로 사용하며, PointCloud2 projection도 기본으로 실행하지 않는다.

## Roughness Anchor

별도의 `roughness_anchor_slam_toolbox_node`는 CSV로 전달된 anchor 후보를 읽고 실제 scan 정합과 graph 일관성 조건을 확인한다. 조건을 만족하는 후보는 pose-graph edge로 반영할 수 있다.

기본 mapping 노드와는 구분된 선택 기능이다. 기존 edge는 기본적으로 교체하지 않으며, innovation frontend와 anchor frontend를 동시에 선택할 수 없도록 구성돼 있다.

## 입출력과 TF 역할

- 입력: `/scan_laserscan`, `/imu/data`, 외부의 `odom -> base_link` TF.
- 출력: `/map` 점유지도와 `map -> odom` 보정 TF.
- 지도와 pose graph를 저장하고 다시 사용할 수 있다.
- Pose-graph exporter로 timestamp, corrected pose, odometric pose 및 edge 정보를 추출할 수 있다.
- `odom -> base_link`는 외부 odometry 추정기가 담당하고, mapping 중의 `map -> odom`은 SLAM이 담당한다.

## 기본 동작과 선택 기능

기본 실행은 custom frontend이며 roughness innovation과 adaptive constraint 처리는 observe 모드다. Near-chain gate, spatial-memory gate, short-passage gate, scan conditioning, reference-chain observer는 자동 활성화하지 않는다.

Live mapping, 호환 bringup, 단순 bag playback, 확장 bag playback 진입점이 있다. 단순 playback의 custom 경로는 SLAM 자체 초기화 후 parameter service 응답을 확인하고 재생을 시작한다. Stock Toolbox 실행 파일도 명시적인 선택으로 사용할 수 있다.

2D SLAM 관련 처리는 `slam_2d_core`와 `slam_2d_bringup`에 있으며, 다른 로봇 노드의 역할은 변경하지 않는다.
