# 2D SLAM 실행

이 패키지는 `slam_2d_core/roughness_aware_slam_toolbox_node`를 직접 실행한다.
기본 `slam_toolbox/sync_slam_toolbox_node`로 전환하지 않는다.
자체 C++ 코드는 `slam_toolbox`의 라이브러리를 사용하며, 노드 이름과 YAML의
최상위 키는 기존 인터페이스인 `slam_toolbox`를 유지한다.

## 빌드와 실행

workspace 루트에서:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select slam_2d_core slam_2d_bringup
source install/setup.bash
ros2 launch slam_2d_bringup slam.launch.py
```

기본 입력은 `/scan_laserscan` (`sensor_msgs/msg/LaserScan`)과 `/imu/data`
(`sensor_msgs/msg/Imu`)이다. LiDAR의 PointCloud2를 직접 받는 launch가 아니므로
LaserScan 변환은 별도 파이프라인에서 제공해야 한다. `odom -> base_link`와
`base_link -> <scan frame>` TF 역시 외부에서 제공해야 한다.
하드웨어 드라이버나 로봇 구동 노드는 이 launch에서 시작하지 않는다.

```bash
ros2 launch slam_2d_bringup slam.launch.py \
  scan_topic:=/scan_laserscan roughness_innovation_imu_topic:=/imu/data
```

기본 설정은 `config/mapper_params.yaml`과
`config/slam_roughness_innovation_params.yaml`이다. `params_file`과
`roughness_innovation_params_file`로 각각 다른 YAML을 지정할 수 있다.
실시간 실행은 `use_sim_time:=false`가 기본이고 bag 재생에는 `true`를 사용한다.

## Bag 재생

다른 로봇/노드와 연결되지 않는 ROS domain에서 실행한다.

```bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=220
ros2 launch slam_2d_bringup slam_from_bag.launch.py bag_file:=/absolute/path/to/bag
```

Bag에는 LaserScan, IMU와 필요한 TF가 있어야 한다. 원본 bag은 수정하지 않는다.
기존 SLAM 결과의 `map -> odom` TF가 포함된 bag은 중복 TF를 만들 수 있으므로,
센서 입력과 odometry TF만 담은 bag을 사용한다. 기본 재생 토픽 목록은 구동 명령을
제외한다. `play_slam_topics_only:=false`로 전체 토픽을 재생할 때는 반드시
실제 로봇과 분리된 환경을 사용한다.
`slam_bringup_launch.py`는 `slam.launch.py`의 별칭이고,
`slam_from_bag_launch.py`는 `slam_from_bag.launch.py`의 별칭이다.
모든 실행 인자는 `ros2 launch slam_2d_bringup <launch 파일> --show-args`로 확인한다.
