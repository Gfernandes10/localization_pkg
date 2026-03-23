#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "../ros_utils/csv_logger.h"

class SimpleKalmanFilter : public rclcpp::Node {
private:
    geometry_msgs::msg::PoseStamped pose_;
    nav_msgs::msg::Odometry filtered_pose_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr filtered_pose_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr natnet_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    Eigen::MatrixXd F_est_;
    Eigen::MatrixXd G_est_;
    Eigen::MatrixXd H_est_;
    Eigen::MatrixXd Q_est_;
    Eigen::MatrixXd R_est_;
    Eigen::VectorXd x_est_;
    Eigen::VectorXd z_;
    Eigen::MatrixXd P_est_;

    std::unique_ptr<CSVLogger> csv_logger_filtered_pose_;
    std::unique_ptr<CSVLogger> csv_logger_ground_truth_;

    std::string natnet_topic_ = "/natnet_ros/Bebop1/pose";
    std::string filtered_pose_topic_ = "/filtered_pose";
    bool has_measurement_ = false;
    bool has_new_measurement_ = false;
    bool filter_initialized_ = false;
    rclcpp::Time last_measurement_time_;
    rclcpp::Time last_measurement_arrival_time_;
    rclcpp::Time last_predict_time_;

    double measurement_timeout_sec_ = 0.5;
    double timer_period_sec_ = 0.01;
    double sigma_a_ = 30.0;
    double measurement_var_ = 0.1;

    rclcpp::Time resolveMeasurementTime(const geometry_msgs::msg::PoseStamped &msg) const {
        const auto &stamp = msg.header.stamp;
        if (stamp.sec != 0 || stamp.nanosec != 0) {
            return rclcpp::Time(stamp, this->get_clock()->get_clock_type());
        }
        return this->now();
    }

    void initializeFilterMatrices() {
        F_est_ = Eigen::MatrixXd::Identity(12, 12);
        G_est_ = Eigen::MatrixXd::Zero(12, 6);
        H_est_ = Eigen::MatrixXd::Zero(6, 12);
        Q_est_ = Eigen::MatrixXd::Zero(12, 12);
        R_est_ = Eigen::MatrixXd::Identity(6, 6) * measurement_var_;
        x_est_ = Eigen::VectorXd::Zero(12);
        z_ = Eigen::VectorXd::Zero(6);
        P_est_ = Eigen::MatrixXd::Identity(12, 12);

        H_est_(0, 0) = 1;   // x
        H_est_(1, 4) = 1;   // y
        H_est_(2, 8) = 1;   // z
        H_est_(3, 6) = 1;   // roll
        H_est_(4, 2) = 1;   // pitch
        H_est_(5, 10) = 1;  // yaw

        updateDiscreteModel(timer_period_sec_);
    }

    void updateDiscreteModel(double dt) {
        if (!std::isfinite(dt) || dt <= 0.0) {
            dt = timer_period_sec_;
        }

        F_est_.setIdentity();
        G_est_.setZero();

        for (int i = 0; i < 6; ++i) {
            F_est_(2 * i, 2 * i + 1) = dt;
            G_est_(2 * i, i) = 0.5 * dt * dt;
            G_est_(2 * i + 1, i) = dt;
        }

        Q_est_ = G_est_ * Eigen::MatrixXd::Identity(6, 6) * std::pow(sigma_a_, 2) * G_est_.transpose();
    }

    void initializeStateFromMeasurement() {
        x_est_.setZero();
        x_est_(0) = z_(0);   // x
        x_est_(4) = z_(1);   // y
        x_est_(8) = z_(2);   // z
        x_est_(6) = z_(3);   // roll
        x_est_(2) = z_(4);   // pitch
        x_est_(10) = z_(5);  // yaw
        P_est_ = Eigen::MatrixXd::Identity(12, 12);
    }

    void natnetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        pose_ = *msg;
        const rclcpp::Time arrival_time = this->now();
        const rclcpp::Time measurement_time = resolveMeasurementTime(*msg);
        last_measurement_arrival_time_ = arrival_time;
        last_measurement_time_ = measurement_time;
        has_measurement_ = true;
        has_new_measurement_ = true;

        tf2::Quaternion quaternion;
        tf2::fromMsg(pose_.pose.orientation, quaternion);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
        const double yaw_deg = yaw * 180.0 / M_PI;

        z_(0) = pose_.pose.position.x;
        z_(1) = pose_.pose.position.y;
        z_(2) = pose_.pose.position.z;
        z_(3) = roll;
        z_(4) = pitch;
        z_(5) = yaw;

        if (!filter_initialized_) {
            initializeStateFromMeasurement();
            last_predict_time_ = measurement_time;
            filter_initialized_ = true;
        }

        std::vector<std::variant<std::string, double>> data_ground_truth = {
            std::to_string(measurement_time.seconds()),
            natnet_topic_,
            pose_.pose.position.x,
            0.0,
            pitch,
            0.0,
            pose_.pose.position.y,
            0.0,
            roll,
            0.0,
            pose_.pose.position.z,
            0.0,
            yaw,
            yaw_deg,
            0.0,
            quaternion.x(),
            quaternion.y(),
            quaternion.z(),
            quaternion.w()};
        csv_logger_ground_truth_->writeCSV(data_ground_truth);
    }

    void predictTo(const rclcpp::Time &target_time) {
        const double predict_dt = (target_time - last_predict_time_).seconds();
        if (!std::isfinite(predict_dt) || predict_dt <= 0.0) {
            return;
        }

        updateDiscreteModel(predict_dt);
        x_est_ = F_est_ * x_est_;
        P_est_ = F_est_ * P_est_ * F_est_.transpose() + Q_est_;
        last_predict_time_ = target_time;
    }

    void updateWithLatestMeasurement() {
        const Eigen::MatrixXd S = H_est_ * P_est_ * H_est_.transpose() + R_est_;
        const Eigen::MatrixXd K = P_est_ * H_est_.transpose() * S.inverse();
        x_est_ = x_est_ + K * (z_ - H_est_ * x_est_);
        P_est_ = (Eigen::MatrixXd::Identity(12, 12) - K * H_est_) * P_est_;
        has_new_measurement_ = false;
    }

    void publishFilteredPose(const rclcpp::Time &stamp) {
        const double x = x_est_(0);
        const double dx = x_est_(1);
        const double pitch = x_est_(2);
        const double dpitch = x_est_(3);
        const double y = x_est_(4);
        const double dy = x_est_(5);
        const double roll = x_est_(6);
        const double droll = x_est_(7);
        const double z = x_est_(8);
        const double dz = x_est_(9);
        const double yaw = x_est_(10);
        const double yaw_deg = yaw * 180.0 / M_PI;
        const double dyaw = x_est_(11);

        filtered_pose_.header.stamp = stamp;
        filtered_pose_.pose.pose.position.x = x;
        filtered_pose_.pose.pose.position.y = y;
        filtered_pose_.pose.pose.position.z = z;
        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        filtered_pose_.pose.pose.orientation = tf2::toMsg(q);
        filtered_pose_.twist.twist.linear.x = dx;
        filtered_pose_.twist.twist.linear.y = dy;
        filtered_pose_.twist.twist.linear.z = dz;
        filtered_pose_.twist.twist.angular.x = droll;
        filtered_pose_.twist.twist.angular.y = dpitch;
        filtered_pose_.twist.twist.angular.z = dyaw;
        filtered_pose_pub_->publish(filtered_pose_);

        std::vector<std::variant<std::string, double>> data_filtered = {
            std::to_string(stamp.seconds()),
            filtered_pose_topic_,
            x,
            dx,
            pitch,
            dpitch,
            y,
            dy,
            roll,
            droll,
            z,
            dz,
            yaw,
            yaw_deg,
            dyaw,
            filtered_pose_.pose.pose.orientation.x,
            filtered_pose_.pose.pose.orientation.y,
            filtered_pose_.pose.pose.orientation.z,
            filtered_pose_.pose.pose.orientation.w};
        csv_logger_filtered_pose_->writeCSV(data_filtered);
    }

    void applyKalmanFilterAndPublish() {
        if (!has_measurement_ || !filter_initialized_) {
            return;
        }

        const rclcpp::Time now = this->now();
        const double elapsed = (now - last_measurement_arrival_time_).seconds();
        if (elapsed > measurement_timeout_sec_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "No pose measurement for %.3fs (timeout %.3fs). Skipping filtered_pose publish.",
                elapsed,
                measurement_timeout_sec_);
            return;
        }

        if (has_new_measurement_) {
            predictTo(last_measurement_time_);
            updateWithLatestMeasurement();
        }

        publishFilteredPose(now);
    }

public:
    SimpleKalmanFilter() : Node("kalman_filter_node") {
        this->declare_parameter<double>("measurement_timeout_sec", measurement_timeout_sec_);
        this->declare_parameter<double>("timer_period_sec", timer_period_sec_);
        this->declare_parameter<double>("sigma_a", sigma_a_);
        this->declare_parameter<double>("measurement_var", measurement_var_);
        this->get_parameter("measurement_timeout_sec", measurement_timeout_sec_);
        this->get_parameter("timer_period_sec", timer_period_sec_);
        this->get_parameter("sigma_a", sigma_a_);
        this->get_parameter("measurement_var", measurement_var_);

        initializeFilterMatrices();
        last_predict_time_ = this->now();

        const char *workspace_name = std::getenv("MY_WORKSPACE_NAME");
        if (workspace_name == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Environment variable MY_WORKSPACE_NAME is not set.");
            throw std::runtime_error("MY_WORKSPACE_NAME not set");
        }

        std::vector<std::string> header = {
            "timestamp",
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
            "yaw_deg",
            "dyaw",
            "qx",
            "qy",
            "qz",
            "qw"};
        csv_logger_filtered_pose_ = std::make_unique<CSVLogger>(workspace_name, "localization_pkg", "filtered_pose", header);
        csv_logger_ground_truth_ = std::make_unique<CSVLogger>(workspace_name, "localization_pkg", "ground_truth", header);

        filtered_pose_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(filtered_pose_topic_, 10);
        natnet_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            natnet_topic_,
            10,
            std::bind(&SimpleKalmanFilter::natnetCallback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(timer_period_sec_),
            std::bind(&SimpleKalmanFilter::applyKalmanFilterAndPublish, this));

        RCLCPP_INFO(
            this->get_logger(),
            "Simple Kalman initialized with timer_period_sec = %.3f, measurement_timeout_sec = %.3f, sigma_a = %.3f, measurement_var = %.3f",
            timer_period_sec_,
            measurement_timeout_sec_,
            sigma_a_,
            measurement_var_);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleKalmanFilter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
