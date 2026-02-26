#pragma once

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>
#include <chrono>

namespace vloc {
class ImagePublisherWorker {
public:
    ImagePublisherWorker(const rclcpp::Node::SharedPtr& node,
                         const std::string& topic = "debug_image",
                         int queue_size = 1,
                         double max_fps = 30.0)
        : node_(node),
          topic_(topic),
          queue_(queue_size),
          target_period_(max_fps > 0 ? 1.0 / max_fps : 0.0) {}

    void start() {
        if (running_.exchange(true)) return;
        pub_ = node_->create_publisher<sensor_msgs::msg::Image>(topic_, queue_);
        worker_ = std::thread(&ImagePublisherWorker::loop, this);
    }

    void stop() {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        pub_.reset();
    }

    void post(const cv::Mat& img, const rclcpp::Time& stamp = rclcpp::Time(0, 0, RCL_ROS_TIME)) {
        std::lock_guard<std::mutex> lk(mtx_);
        latest_ = img.clone();
        stamp_ = stamp;
        has_frame_ = true;
        cv_.notify_one();
    }

    void setMaxFps(double fps) { target_period_ = fps > 0 ? 1.0 / fps : 0.0; }
    void setTopic(const std::string& t) { topic_ = t; }

private:
    static std::string guessEncoding(const cv::Mat& m) {
        if (m.type() == CV_8UC1) return "mono8";
        if (m.type() == CV_8UC3) return "bgr8";
        return m.channels() == 1 ? "mono8" : "bgr8";
    }

    void loop() {
        using clock = std::chrono::steady_clock;
        auto next_tick = clock::now();
        while (running_) {
            if (target_period_ > 0) {
                next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(target_period_));
            }

            cv::Mat frame;
            rclcpp::Time t(0, 0, RCL_ROS_TIME);
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_for(lk, std::chrono::milliseconds(5), [&] { return !running_ || has_frame_; });
                if (!running_) break;
                if (has_frame_) {
                    frame = latest_;
                    t = stamp_;
                    has_frame_ = false;
                }
            }

            if (!frame.empty() && pub_) {
                std_msgs::msg::Header h;
                h.stamp = t.nanoseconds() == 0 ? node_->now() : t;
                auto msg = cv_bridge::CvImage(h, guessEncoding(frame), frame).toImageMsg();
                pub_->publish(*msg);
            }

            if (target_period_ > 0) std::this_thread::sleep_until(next_tick);
        }
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::string topic_;
    int queue_ = 1;

    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    cv::Mat latest_;
    rclcpp::Time stamp_{0, 0, RCL_ROS_TIME};
    bool has_frame_ = false;
    std::atomic<bool> running_{false};
    double target_period_ = 0.0;
};
}  // namespace vloc
