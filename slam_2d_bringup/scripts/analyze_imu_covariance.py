#!/usr/bin/env python3
"""
IMU 센서 데이터 공분산 분석 도구.

실시간으로 IMU 토픽을 구독하여 공분산 값을 계산합니다.

사용법:
    python3 analyze_imu_covariance.py [--topic /imu/data_raw]
        [--duration 30] [--output output.yaml]
"""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import yaml


class IMUCovarianceAnalyzer(Node):
    def __init__(self, topic_name, duration, output_file):
        super().__init__('imu_covariance_analyzer')

        self.topic_name = topic_name
        self.duration = duration
        self.output_file = output_file

        # 데이터 저장용
        self.orientations = []  # [roll, pitch, yaw]
        self.angular_velocities = []  # [vroll, vpitch, vyaw]
        self.linear_accelerations = []  # [ax, ay, az]

        # Quaternion to Euler 변환용
        self.start_time = None
        self.collection_started = False

        # 구독자 생성
        self.subscription = self.create_subscription(
            Imu,
            topic_name,
            self.imu_callback,
            10
        )

        self.get_logger().info('IMU 공분산 분석 시작')
        self.get_logger().info(f'토픽: {topic_name}')
        self.get_logger().info(f'수집 시간: {duration}초')
        self.get_logger().info('센서를 정지 상태로 두고 데이터를 수집합니다...')

    def quaternion_to_euler(self, x, y, z, w):
        """Quaternion을 Euler angle (roll, pitch, yaw)로 변환합니다."""
        import math

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def imu_callback(self, msg):
        """IMU 메시지 콜백입니다."""
        if not self.collection_started:
            self.start_time = time.time()
            self.collection_started = True
            self.get_logger().info('데이터 수집 시작...')

        elapsed = time.time() - self.start_time

        if elapsed > self.duration:
            self.analyze_and_save()
            rclpy.shutdown()
            return

        # Orientation (quaternion to euler)
        roll, pitch, yaw = self.quaternion_to_euler(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
        self.orientations.append([roll, pitch, yaw])

        # Angular velocity
        self.angular_velocities.append([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Linear acceleration
        self.linear_accelerations.append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # 진행 상황 출력 (1초마다)
        if len(self.orientations) % 100 == 0:  # 약 1초마다 (100Hz 가정)
            remaining = self.duration - elapsed
            self.get_logger().info(
                f'수집 중... ({elapsed:.1f}/{self.duration}초, '
                f'남은 시간: {remaining:.1f}초)'
            )

    def analyze_and_save(self):
        """수집된 데이터를 분석하고 결과를 저장합니다."""
        self.get_logger().info('\n=== 데이터 분석 시작 ===')

        if len(self.orientations) == 0:
            self.get_logger().error('수집된 데이터가 없습니다!')
            return

        # NumPy 배열로 변환
        orientations = np.array(self.orientations)
        angular_velocities = np.array(self.angular_velocities)
        linear_accelerations = np.array(self.linear_accelerations)

        # 공분산 계산
        orientation_cov = np.cov(orientations.T)
        angular_velocity_cov = np.cov(angular_velocities.T)
        linear_acceleration_cov = np.cov(linear_accelerations.T)

        # 대각선 요소만 추출 (분산)
        orientation_variance = np.diag(orientation_cov)
        angular_velocity_variance = np.diag(angular_velocity_cov)
        linear_acceleration_variance = np.diag(linear_acceleration_cov)

        # 통계 정보
        orientation_mean = np.mean(orientations, axis=0)
        angular_velocity_mean = np.mean(angular_velocities, axis=0)
        linear_acceleration_mean = np.mean(linear_accelerations, axis=0)

        # 결과 출력
        self.get_logger().info(f'\n수집된 샘플 수: {len(self.orientations)}')
        self.get_logger().info('\n=== Orientation (Roll, Pitch, Yaw) ===')
        self.get_logger().info(
            f'평균: [{orientation_mean[0]:.6f}, {orientation_mean[1]:.6f}, '
            f'{orientation_mean[2]:.6f}] rad'
        )
        self.get_logger().info(
            f'평균 (도): [{np.degrees(orientation_mean[0]):.2f}, '
            f'{np.degrees(orientation_mean[1]):.2f}, '
            f'{np.degrees(orientation_mean[2]):.2f}] deg'
        )
        self.get_logger().info(
            f'분산: [{orientation_variance[0]:.6f}, '
            f'{orientation_variance[1]:.6f}, '
            f'{orientation_variance[2]:.6f}] rad²'
        )
        self.get_logger().info(
            f'표준편차: [{np.sqrt(orientation_variance[0]):.6f}, '
            f'{np.sqrt(orientation_variance[1]):.6f}, '
            f'{np.sqrt(orientation_variance[2]):.6f}] rad'
        )
        self.get_logger().info(
            f'표준편차 (도): '
            f'[{np.degrees(np.sqrt(orientation_variance[0])):.2f}, '
            f'{np.degrees(np.sqrt(orientation_variance[1])):.2f}, '
            f'{np.degrees(np.sqrt(orientation_variance[2])):.2f}] deg'
        )

        self.get_logger().info(
            '\n=== Angular Velocity (Vroll, Vpitch, Vyaw) ==='
        )
        self.get_logger().info(
            f'평균: [{angular_velocity_mean[0]:.6f}, '
            f'{angular_velocity_mean[1]:.6f}, '
            f'{angular_velocity_mean[2]:.6f}] rad/s'
        )
        self.get_logger().info(
            f'분산: [{angular_velocity_variance[0]:.6f}, '
            f'{angular_velocity_variance[1]:.6f}, '
            f'{angular_velocity_variance[2]:.6f}] (rad/s)²'
        )
        self.get_logger().info(
            f'표준편차: [{np.sqrt(angular_velocity_variance[0]):.6f}, '
            f'{np.sqrt(angular_velocity_variance[1]):.6f}, '
            f'{np.sqrt(angular_velocity_variance[2]):.6f}] rad/s'
        )

        self.get_logger().info(
            '\n=== Linear Acceleration (Ax, Ay, Az) ==='
        )
        self.get_logger().info(
            f'평균: [{linear_acceleration_mean[0]:.6f}, '
            f'{linear_acceleration_mean[1]:.6f}, '
            f'{linear_acceleration_mean[2]:.6f}] m/s²'
        )
        self.get_logger().info(
            f'분산: [{linear_acceleration_variance[0]:.6f}, '
            f'{linear_acceleration_variance[1]:.6f}, '
            f'{linear_acceleration_variance[2]:.6f}] (m/s²)²'
        )
        self.get_logger().info(
            f'표준편차: [{np.sqrt(linear_acceleration_variance[0]):.6f}, '
            f'{np.sqrt(linear_acceleration_variance[1]):.6f}, '
            f'{np.sqrt(linear_acceleration_variance[2]):.6f}] m/s²'
        )

        # EKF 설정용 추천 값 생성
        # 공분산 행렬은 9x9 형태이지만, 대각선 요소만 사용
        recommended_config = {
            'imu_covariance_recommendations': {
                'orientation_covariance': {
                    'roll': float(orientation_variance[0]),
                    'pitch': float(orientation_variance[1]),
                    'yaw': float(orientation_variance[2]),
                    'note': 'rad² 단위. EKF 설정 파일의 imu0_config에서 yaw를 false로 설정했다면 yaw는 0.0 사용'
                },
                'angular_velocity_covariance': {
                    'vroll': float(angular_velocity_variance[0]),
                    'vpitch': float(angular_velocity_variance[1]),
                    'vyaw': float(angular_velocity_variance[2]),
                    'note': '(rad/s)² 단위'
                },
                'linear_acceleration_covariance': {
                    'ax': float(linear_acceleration_variance[0]),
                    'ay': float(linear_acceleration_variance[1]),
                    'az': float(linear_acceleration_variance[2]),
                    'note': '(m/s²)² 단위'
                },
                'statistics': {
                    'samples_collected': int(len(self.orientations)),
                    'duration_seconds': float(self.duration),
                    'sampling_rate_hz': float(len(self.orientations) / self.duration)
                }
            }
        }

        # Python 코드로 사용할 수 있는 형태
        python_code = f"""
# IMU 드라이버에 추가할 공분산 설정 코드
# wit_ros2_imu.py의 imu_data() 메서드에 추가

# Orientation covariance (rad²)
self.imu_msg.orientation_covariance[0] = {orientation_variance[0]:.6f}  # roll
self.imu_msg.orientation_covariance[4] = {orientation_variance[1]:.6f}  # pitch
self.imu_msg.orientation_covariance[8] = {orientation_variance[2]:.6f}  # yaw (사용 안함시 0.0)

# Angular velocity covariance (rad²/s²)
self.imu_msg.angular_velocity_covariance[0] = {angular_velocity_variance[0]:.6f}  # vroll
self.imu_msg.angular_velocity_covariance[4] = {angular_velocity_variance[1]:.6f}  # vpitch
self.imu_msg.angular_velocity_covariance[8] = {angular_velocity_variance[2]:.6f}  # vyaw

# Linear acceleration covariance (m²/s⁴)
self.imu_msg.linear_acceleration_covariance[0] = {linear_acceleration_variance[0]:.6f}  # ax
self.imu_msg.linear_acceleration_covariance[4] = {linear_acceleration_variance[1]:.6f}  # ay
self.imu_msg.linear_acceleration_covariance[8] = {linear_acceleration_variance[2]:.6f}  # az
"""

        # 파일 저장
        if self.output_file:
            with open(self.output_file, 'w') as f:
                yaml.dump(recommended_config, f, default_flow_style=False, allow_unicode=True)

            code_file = self.output_file.replace('.yaml', '_code.py')
            with open(code_file, 'w') as f:
                f.write(python_code)

            self.get_logger().info('\n결과 저장 완료:')
            self.get_logger().info(f'  - {self.output_file} (YAML 형식)')
            self.get_logger().info(f'  - {code_file} (Python 코드)')
        else:
            self.get_logger().info('\n=== 추천 공분산 값 (Python 코드) ===')
            self.get_logger().info(python_code)

        self.get_logger().info('\n=== 분석 완료 ===')


def main():
    parser = argparse.ArgumentParser(description='IMU 센서 공분산 분석 도구')
    parser.add_argument('--topic', type=str, default='/imu/data_raw',
                        help='IMU 토픽 이름 (기본값: /imu/data_raw)')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='데이터 수집 시간 (초, 기본값: 30초)')
    parser.add_argument('--output', type=str, default='imu_covariance_analysis.yaml',
                        help='출력 파일 이름 (기본값: imu_covariance_analysis.yaml)')

    args = parser.parse_args()

    rclpy.init()

    analyzer = IMUCovarianceAnalyzer(
        topic_name=args.topic,
        duration=args.duration,
        output_file=args.output
    )

    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        analyzer.get_logger().info('사용자에 의해 중단됨')
        analyzer.analyze_and_save()
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
