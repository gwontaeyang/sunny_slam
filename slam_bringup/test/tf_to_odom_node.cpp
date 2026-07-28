#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

class TfToOdomNode : public rclcpp::Node
{
public:
  TfToOdomNode() : Node("tf_to_odom_node")
  {
    RCLCPP_INFO(this->get_logger(), "tf_to_odom_node 시작");

    // 파라미터 선언
    this->declare_parameter<std::string>("parent_frame", "odom");
    this->declare_parameter<std::string>("child_frame", "base_link");
    this->declare_parameter<std::string>("odom_topic_name", "odom");
    this->declare_parameter<double>("publish_rate", 50.0);

    // 파라미터 가져오기
    std::string parent_frame, child_frame, odom_topic_name;
    double publish_rate;

    this->get_parameter("parent_frame", parent_frame);
    this->get_parameter("child_frame", child_frame);
    this->get_parameter("odom_topic_name", odom_topic_name);
    this->get_parameter("publish_rate", publish_rate);

    RCLCPP_INFO(this->get_logger(), "===== Configuration =====");
    RCLCPP_INFO(this->get_logger(), "parent_frame: %s", parent_frame.c_str());
    RCLCPP_INFO(this->get_logger(), "child_frame: %s", child_frame.c_str());
    RCLCPP_INFO(this->get_logger(), "odom_topic_name: %s", odom_topic_name.c_str());
    RCLCPP_INFO(this->get_logger(), "publish_rate: %.2f Hz", publish_rate);

    // TF 버퍼와 리스너 초기화
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // Odometry 발행자 생성
    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic_name, 10);

    // 타이머로 주기적으로 TF를 읽어서 Odometry 발행
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
      [this, parent_frame, child_frame]() {
        this->publish_odom_from_tf(parent_frame, child_frame);
      }
    );
  }

private:
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  void publish_odom_from_tf(const std::string& parent_frame, const std::string& child_frame)
  {
    try {
      // TF 조회 (최신 transform)
      geometry_msgs::msg::TransformStamped transform;
      // 최신 transform을 가져오기 위해 rclcpp::Time(0) 사용
      transform = tf_buffer_->lookupTransform(
        parent_frame, child_frame,
        rclcpp::Time(0)  // 최신 transform 요청
      );

      // Odometry 메시지 생성
      nav_msgs::msg::Odometry odom_msg;
      odom_msg.header.stamp = transform.header.stamp;      odom_msg.header.frame_id = parent_frame;
      odom_msg.child_frame_id = child_frame;

      // Pose 설정
      odom_msg.pose.pose.position.x = transform.transform.translation.x;
      odom_msg.pose.pose.position.y = transform.transform.translation.y;
      odom_msg.pose.pose.position.z = transform.transform.translation.z;
      odom_msg.pose.pose.orientation = transform.transform.rotation;

      // Covariance 설정 (기본값, 필요시 파라미터로 설정 가능)
      // Position covariance
      odom_msg.pose.covariance[0] = 0.01;  // x
      odom_msg.pose.covariance[7] = 0.01;  // y
      odom_msg.pose.covariance[14] = 0.01; // z
      odom_msg.pose.covariance[21] = 0.01; // roll
      odom_msg.pose.covariance[28] = 0.01; // pitch
      odom_msg.pose.covariance[35] = 0.01; // yaw

      // TODO: Twist 계산
      odom_msg.twist.twist.linear.x = transform.transform.translation.x;
      odom_msg.twist.twist.linear.y = transform.transform.translation.y;
      odom_msg.twist.twist.linear.z = transform.transform.translation.z;
      odom_msg.twist.twist.angular.x = transform.transform.rotation.x;
      odom_msg.twist.twist.angular.y = transform.transform.rotation.y;
      odom_msg.twist.twist.angular.z = transform.transform.rotation.z;

      // Twist covariance
      for (int i = 0; i < 36; i++) {
        odom_msg.twist.covariance[i] = 0.0;
      }

      // 발행
      odom_publisher_->publish(odom_msg);

    } catch (tf2::TransformException &ex) {
      // TF를 찾을 수 없는 경우 (일반적으로 처음 시작할 때 발생)
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        5000,
        "TF 변환 실패: %s -> %s: %s",
        parent_frame.c_str(),
        child_frame.c_str(),
        ex.what()
      );
    }
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TfToOdomNode>());
  rclcpp::shutdown();
  return 0;
}
