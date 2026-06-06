#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <deque>
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
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
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
    std::string command_topic_ = "/tello/cmd_vel";
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
    bool publish_prediction_without_measurement_ = true;
    double max_prediction_duration_sec_ = 2.0;
    bool use_control_input_model_ = false;
    double control_input_timeout_sec_ = 0.25;
    double control_input_delay_sec_ = 0.0;
    double control_model_max_step_sec_ = 0.02;
    double gamma1_ = 3.75;
    double gamma2_ = -1.10;
    double gamma3_ = 3.75;
    double gamma4_ = -1.10;
    double gamma5_ = 2.68;
    double gamma6_ = -0.75;
    double gamma7_ = 1.42;
    double gamma8_ = -2.06;

    struct TimedCommand {
        rclcpp::Time stamp;
        geometry_msgs::msg::Twist cmd;
    };
    std::deque<TimedCommand> command_buffer_;
    std::size_t command_buffer_max_size_ = 1000;

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

    void cmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        TimedCommand timed_cmd;
        timed_cmd.stamp = this->now();
        timed_cmd.cmd = *msg;
        command_buffer_.push_back(timed_cmd);
        while (command_buffer_.size() > command_buffer_max_size_) {
            command_buffer_.pop_front();
        }
    }

    bool getCommandForPrediction(
        const rclcpp::Time &target_time,
        geometry_msgs::msg::Twist &cmd) const
    {
        if (!use_control_input_model_ || command_buffer_.empty()) {
            return false;
        }

        const rclcpp::Time effective_time =
            target_time - rclcpp::Duration::from_seconds(control_input_delay_sec_);

        for (auto it = command_buffer_.rbegin(); it != command_buffer_.rend(); ++it) {
            if (it->stamp <= effective_time) {
                const double age = (effective_time - it->stamp).seconds();
                if (std::isfinite(age) && age <= control_input_timeout_sec_) {
                    cmd = it->cmd;
                    return true;
                }
                return false;
            }
        }

        return false;
    }

    void predictStateWithControlModel(
        double dt,
        const geometry_msgs::msg::Twist &cmd)
    {
        if (!std::isfinite(dt) || dt <= 0.0) {
            return;
        }

        const double max_step =
            (std::isfinite(control_model_max_step_sec_) && control_model_max_step_sec_ > 0.0)
                ? control_model_max_step_sec_
                : timer_period_sec_;
        const int steps = std::max(1, static_cast<int>(std::ceil(dt / max_step)));
        const double h = dt / static_cast<double>(steps);

        for (int step = 0; step < steps; ++step) {
            const double yaw = x_est_(10);
            const double c = std::cos(yaw);
            const double s = std::sin(yaw);

            const double vx_world = x_est_(1);
            const double vy_world = x_est_(5);
            const double vz = x_est_(9);
            const double vyaw = x_est_(11);

            const double vx_body = c * vx_world + s * vy_world;
            const double vy_body = -s * vx_world + c * vy_world;

            const double ax_body = gamma2_ * vx_body + gamma1_ * cmd.linear.x;
            const double ay_body = gamma4_ * vy_body + gamma3_ * cmd.linear.y;
            const double ax_world = c * ax_body - s * ay_body;
            const double ay_world = s * ax_body + c * ay_body;
            const double az = gamma6_ * vz + gamma5_ * cmd.linear.z;
            const double ayaw = gamma8_ * vyaw + gamma7_ * cmd.angular.z;

            x_est_(0) += vx_world * h + 0.5 * ax_world * h * h;
            x_est_(1) += ax_world * h;
            x_est_(4) += vy_world * h + 0.5 * ay_world * h * h;
            x_est_(5) += ay_world * h;
            x_est_(8) += vz * h + 0.5 * az * h * h;
            x_est_(9) += az * h;
            x_est_(10) += vyaw * h + 0.5 * ayaw * h * h;
            x_est_(11) += ayaw * h;

            // Roll and pitch are not commanded by this model; keep their constant-velocity prediction.
            x_est_(2) += x_est_(3) * h;
            x_est_(6) += x_est_(7) * h;
        }
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
        geometry_msgs::msg::Twist cmd;
        if (getCommandForPrediction(target_time, cmd)) {
            predictStateWithControlModel(predict_dt, cmd);
        } else {
            x_est_ = F_est_ * x_est_;
        }
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
            if (!publish_prediction_without_measurement_ || elapsed > max_prediction_duration_sec_) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    2000,
                    "No pose measurement for %.3fs (timeout %.3fs, max prediction %.3fs). Skipping filtered_pose publish.",
                    elapsed,
                    measurement_timeout_sec_,
                    max_prediction_duration_sec_);
                return;
            }

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "No pose measurement for %.3fs (timeout %.3fs). Publishing predicted filtered_pose.",
                elapsed,
                measurement_timeout_sec_);
        }

        if (has_new_measurement_) {
            predictTo(last_measurement_time_);
            updateWithLatestMeasurement();
        }

        if (publish_prediction_without_measurement_) {
            predictTo(now);
        }

        publishFilteredPose(now);
    }

public:
    SimpleKalmanFilter() : Node("kalman_filter_node") {
        this->declare_parameter<double>("measurement_timeout_sec", measurement_timeout_sec_);
        this->declare_parameter<double>("timer_period_sec", timer_period_sec_);
        this->declare_parameter<double>("sigma_a", sigma_a_);
        this->declare_parameter<double>("measurement_var", measurement_var_);
        this->declare_parameter<bool>("publish_prediction_without_measurement", publish_prediction_without_measurement_);
        this->declare_parameter<double>("max_prediction_duration_sec", max_prediction_duration_sec_);
        this->declare_parameter<std::string>("command_topic", command_topic_);
        this->declare_parameter<bool>("use_control_input_model", use_control_input_model_);
        this->declare_parameter<double>("control_input_timeout_sec", control_input_timeout_sec_);
        this->declare_parameter<double>("control_input_delay_sec", control_input_delay_sec_);
        this->declare_parameter<double>("control_model_max_step_sec", control_model_max_step_sec_);
        this->declare_parameter<double>("control_model_gamma1", gamma1_);
        this->declare_parameter<double>("control_model_gamma2", gamma2_);
        this->declare_parameter<double>("control_model_gamma3", gamma3_);
        this->declare_parameter<double>("control_model_gamma4", gamma4_);
        this->declare_parameter<double>("control_model_gamma5", gamma5_);
        this->declare_parameter<double>("control_model_gamma6", gamma6_);
        this->declare_parameter<double>("control_model_gamma7", gamma7_);
        this->declare_parameter<double>("control_model_gamma8", gamma8_);
        this->get_parameter("measurement_timeout_sec", measurement_timeout_sec_);
        this->get_parameter("timer_period_sec", timer_period_sec_);
        this->get_parameter("sigma_a", sigma_a_);
        this->get_parameter("measurement_var", measurement_var_);
        this->get_parameter("publish_prediction_without_measurement", publish_prediction_without_measurement_);
        this->get_parameter("max_prediction_duration_sec", max_prediction_duration_sec_);
        this->get_parameter("command_topic", command_topic_);
        this->get_parameter("use_control_input_model", use_control_input_model_);
        this->get_parameter("control_input_timeout_sec", control_input_timeout_sec_);
        this->get_parameter("control_input_delay_sec", control_input_delay_sec_);
        this->get_parameter("control_model_max_step_sec", control_model_max_step_sec_);
        this->get_parameter("control_model_gamma1", gamma1_);
        this->get_parameter("control_model_gamma2", gamma2_);
        this->get_parameter("control_model_gamma3", gamma3_);
        this->get_parameter("control_model_gamma4", gamma4_);
        this->get_parameter("control_model_gamma5", gamma5_);
        this->get_parameter("control_model_gamma6", gamma6_);
        this->get_parameter("control_model_gamma7", gamma7_);
        this->get_parameter("control_model_gamma8", gamma8_);

        if (max_prediction_duration_sec_ < measurement_timeout_sec_) {
            RCLCPP_WARN(
                this->get_logger(),
                "max_prediction_duration_sec %.3f is smaller than measurement_timeout_sec %.3f. Clamping max_prediction_duration_sec to measurement_timeout_sec.",
                max_prediction_duration_sec_,
                measurement_timeout_sec_);
            max_prediction_duration_sec_ = measurement_timeout_sec_;
        }

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
        cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            command_topic_,
            10,
            std::bind(&SimpleKalmanFilter::cmdCallback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(timer_period_sec_),
            std::bind(&SimpleKalmanFilter::applyKalmanFilterAndPublish, this));

        RCLCPP_INFO(
            this->get_logger(),
            "Simple Kalman initialized with timer_period_sec = %.3f, measurement_timeout_sec = %.3f, sigma_a = %.3f, measurement_var = %.3f, publish_prediction_without_measurement = %s, max_prediction_duration_sec = %.3f, use_control_input_model = %s, command_topic = %s",
            timer_period_sec_,
            measurement_timeout_sec_,
            sigma_a_,
            measurement_var_,
            publish_prediction_without_measurement_ ? "true" : "false",
            max_prediction_duration_sec_,
            use_control_input_model_ ? "true" : "false",
            command_topic_.c_str());
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleKalmanFilter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
