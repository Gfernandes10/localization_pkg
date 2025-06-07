#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include "../ros_utils/csv_logger.h"

class SimpleKalmanFilter : public rclcpp::Node {
    private:
        // Class members
        geometry_msgs::msg::Quaternion quat;
        nav_msgs::msg::Odometry filtered_pose;
        geometry_msgs::msg::PoseStamped pose;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr filtered_pose_pub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr natnet_sub_;
        rclcpp::TimerBase::SharedPtr timer_;
        Eigen::MatrixXd P_;
        Eigen::MatrixXd P_pred_;
        Eigen::MatrixXd F_;
        Eigen::MatrixXd H_;
        Eigen::MatrixXd Q_;
        Eigen::MatrixXd R_;
        Eigen::VectorXd x_pred_;
        Eigen::VectorXd z_;
        Eigen::VectorXd x_;
        Eigen::VectorXd x_odom_;
        std::unique_ptr<CSVLogger> csv_logger_filtered_pose_;
        std::unique_ptr<CSVLogger> csv_logger_ground_truth_;
        std::ostringstream timestamp;
        std::string natnet_topic = "/natnet_ros/Bebop1/pose";
        std::string filtered_pose_topic = "/filtered_pose";

        // Estimated Kalman Parameters
        bool use_odom = false;
        Eigen::MatrixXd F_est;
        Eigen::MatrixXd G_est;
        Eigen::MatrixXd H_est;
        Eigen::MatrixXd Q_est;
        Eigen::MatrixXd R_est;
        Eigen::VectorXd x_est;
        Eigen::MatrixXd P_est;
public:
    SimpleKalmanFilter() : Node("kalman_filter_node") {
        // Initialize Kalman filter matrices
        P_ = Eigen::MatrixXd::Zero(12, 12);
        P_pred_ = Eigen::MatrixXd::Zero(12, 12);
        F_ = Eigen::MatrixXd::Identity(12, 12);
        H_ = Eigen::MatrixXd::Zero(6, 12);
        H_(0, 0) = 1;
        H_(1, 4) = 1;
        H_(2, 8) = 1;
        H_(3, 6) = 1;
        H_(4, 2) = 1;
        H_(5, 10) = 1;
        Q_ = Eigen::MatrixXd::Identity(12, 12) * 0.2;
        R_ = Eigen::MatrixXd::Identity(6, 6);
        x_pred_ = Eigen::VectorXd::Zero(12);
        x_odom_ = Eigen::VectorXd::Zero(12);
        z_ = Eigen::VectorXd::Zero(6);
        x_ = Eigen::VectorXd::Zero(12);

        // Estimated Kalman Filter Parameters
        double Ts = 0.01;
        double sigma_a = 30.0;
        initializeEstimatedKalmanParameters(Ts, sigma_a);

        // Initialize loggers
        const char* workspace_name = std::getenv("MY_WORKSPACE_NAME");
        if (workspace_name == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Environment variable MY_WORKSPACE_NAME is not set.");
            throw std::runtime_error("MY_WORKSPACE_NAME not set");
        }
        std::vector<std::string> header = {"timestamp", 
                                            "topic", 
                                            "x", 
                                            "dx", 
                                            "theta", 
                                            "dtheta", 
                                            "y", 
                                            "dy", 
                                            "roll", 
                                            "droll", 
                                            "z", 
                                            "dz", 
                                            "yaw", 
                                            "dyaw", 
                                            "qx", 
                                            "qy", 
                                            "qz", 
                                            "qw"};
        csv_logger_filtered_pose_ = std::make_unique<CSVLogger>(workspace_name, "localization_pkg", "filtered_pose", header);
        csv_logger_ground_truth_ = std::make_unique<CSVLogger>(workspace_name, "localization_pkg", "ground_truth", header);

        // Publishers
        filtered_pose_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(filtered_pose_topic, 10);

        // Subscribers
        natnet_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(natnet_topic, 10, std::bind(&SimpleKalmanFilter::natnetCallback, this, std::placeholders::_1));

        // Timer
        timer_ = this->create_wall_timer(std::chrono::duration<double>(Ts), std::bind(&SimpleKalmanFilter::timerCallback, this));
        RCLCPP_INFO(this->get_logger(), "Timer initialized with Ts = %f", Ts);
    }


    void initializeEstimatedKalmanParameters(double Ts, double sigma) {
        F_est = Eigen::MatrixXd::Identity(12, 12);
        G_est = Eigen::MatrixXd(12, 6);
        H_est = Eigen::MatrixXd::Zero(6, 12);
        Q_est = Eigen::MatrixXd::Zero(12, 12);
        R_est = Eigen::MatrixXd::Identity(6, 6);
        x_est = Eigen::VectorXd::Zero(12);
        P_est = Eigen::MatrixXd::Identity(12, 12);
        for (int i = 0; i < 6; ++i) {
            F_est(2 * i, 2 * i + 1) = Ts;
        }
        for (int i = 0; i < 6; ++i) {
            G_est(2 * i, i) = 0.5 * Ts * Ts;
            G_est(2 * i + 1, i) = Ts;
        }
        H_est(0, 0) = 1;
        H_est(1, 4) = 1;
        H_est(2, 8) = 1;
        H_est(3, 6) = 1;
        H_est(4, 2) = 1;
        H_est(5, 10) = 1;
        Q_est = G_est * G_est.transpose() * sigma * sigma;
        R_est = Eigen::MatrixXd::Identity(6, 6) * 0.1;
    }


    void natnetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        pose = *msg;
        tf2::Quaternion quaternion;
        tf2::fromMsg(pose.pose.orientation, quaternion);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
        double x = pose.pose.position.x;
        double y = pose.pose.position.y;
        double z = pose.pose.position.z;
        z_(0) = x;
        z_(1) = y;
        z_(2) = z;
        z_(3) = roll;
        z_(4) = pitch;
        z_(5) = yaw;
        std::vector<std::variant<std::string, double>> data_ground_truth = {std::to_string(this->now().seconds()), natnet_topic, pose.pose.position.x, 0.0, pitch, 0.0, pose.pose.position.y, 0.0, roll, 0.0, pose.pose.position.z, 0.0, yaw, 0.0, quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()};
        csv_logger_ground_truth_->writeCSV(data_ground_truth);
    }

    void timerCallback() {
        applyKalmanFilterAndPublish();
    }

    void applyKalmanFilterAndPublish() {
        Eigen::VectorXd x_pred_est = F_est * x_est;
        Eigen::MatrixXd P_pred_est = F_est * P_est * F_est.transpose() + Q_est;
        Eigen::MatrixXd K = P_pred_est * H_est.transpose() * (H_est * P_pred_est * H_est.transpose() + R_est).inverse();
        x_est = x_pred_est + K * (z_ - H_est * x_pred_est);
        P_est = (Eigen::MatrixXd::Identity(12, 12) - K * H_est) * P_pred_est;
        double x = x_est(0);
        double dx = x_est(1);
        double pitch = x_est(2);
        double dpitch = x_est(3);
        double y = x_est(4);
        double dy = x_est(5);
        double roll = x_est(6);
        double droll = x_est(7);
        double z = x_est(8);
        double dz = x_est(9);
        double yaw = x_est(10);
        double dyaw = x_est(11);
        filtered_pose.pose.pose.position.x = x;
        filtered_pose.pose.pose.position.y = y;
        filtered_pose.pose.pose.position.z = z;
        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        filtered_pose.pose.pose.orientation = tf2::toMsg(q);
        filtered_pose.twist.twist.linear.x = dx;
        filtered_pose.twist.twist.linear.y = dy;
        filtered_pose.twist.twist.linear.z = dz;
        filtered_pose.twist.twist.angular.x = droll;
        filtered_pose.twist.twist.angular.y = dpitch;
        filtered_pose.twist.twist.angular.z = dyaw;
        filtered_pose_pub_->publish(filtered_pose);
        std::vector<std::variant<std::string, double>> data_filtered = {std::to_string(this->now().seconds()), filtered_pose_topic, x, dx, pitch, dpitch, y, dy, roll, droll, z, dz, yaw, dyaw, filtered_pose.pose.pose.orientation.x, filtered_pose.pose.pose.orientation.y, filtered_pose.pose.pose.orientation.z, filtered_pose.pose.pose.orientation.w};
        csv_logger_filtered_pose_->writeCSV(data_filtered);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleKalmanFilter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
