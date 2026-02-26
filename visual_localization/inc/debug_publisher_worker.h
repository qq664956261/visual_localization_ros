#pragma once

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <string>
#include <chrono>

namespace vloc {

class DebugPublisherWorker {
public:
    using Cloud = pcl::PointCloud<pcl::PointXYZ>;
    using CloudConstPtr = Cloud::ConstPtr;

    explicit DebugPublisherWorker(const rclcpp::Node::SharedPtr& node,
                                  const std::string& image_ns = "debug_image",
                                  const std::string& cloud_ns = "debug_cloud")
        : node_(node), image_ns_(image_ns), cloud_ns_(cloud_ns) {}

    void start() {
        if (running_.exchange(true)) return;
        worker_ = std::thread(&DebugPublisherWorker::loop, this);
    }

    void stop() {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        image_channels_.clear();
        cloud_channels_.clear();
    }

    void addImageChannel(const std::string& tag, double fps = 30.0) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto& ch = image_channels_[tag];
        if (!ch.initialized) {
            ch.topic = imageTopic(tag);
            ch.pub = node_->create_publisher<sensor_msgs::msg::Image>(ch.topic, 1);
            ch.initialized = true;
        }
        ch.fps = fps;
        ch.period = fps > 0 ? 1.0 / fps : 0.0;
    }

    void addCloudChannel(const std::string& tag, double fps = 5.0, const std::string& frame_id = "map") {
        std::lock_guard<std::mutex> lk(mtx_);
        auto& ch = cloud_channels_[tag];
        if (!ch.initialized) {
            ch.topic = cloudTopic(tag);
            ch.pub = node_->create_publisher<sensor_msgs::msg::PointCloud2>(ch.topic, 1);
            ch.initialized = true;
        }
        ch.fps = fps;
        ch.period = fps > 0 ? 1.0 / fps : 0.0;
        ch.frame_id = frame_id;
    }

    void postImage(const std::string& tag, const cv::Mat& img, const rclcpp::Time& stamp) {
        addImageChannel(tag);
        std::lock_guard<std::mutex> lk(mtx_);
        auto& ch = image_channels_[tag];
        ch.latest = img.clone();
        ch.stamp = stamp;
        ch.has_data = true;
        cv_.notify_one();
    }

    void postCloud(const std::string& tag, const CloudConstPtr& cloud, const rclcpp::Time& stamp,
                   const std::string& frame_id = "") {
        addCloudChannel(tag);
        std::lock_guard<std::mutex> lk(mtx_);
        auto& ch = cloud_channels_[tag];
        ch.latest = cloud;
        ch.stamp = stamp;
        if (!frame_id.empty()) ch.frame_id = frame_id;
        ch.has_data = true;
        cv_.notify_one();
    }

private:
    struct ImageCh {
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub;
        std::string topic;
        bool initialized = false;
        cv::Mat latest;
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        bool has_data = false;
        double fps = 30.0;
        double period = 1.0 / 30.0;
        rclcpp::Time last_pub{0, 0, RCL_ROS_TIME};
    };

    struct CloudCh {
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub;
        std::string topic;
        bool initialized = false;
        CloudConstPtr latest;
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        bool has_data = false;
        std::string frame_id = "map";
        double fps = 5.0;
        double period = 1.0 / 5.0;
        rclcpp::Time last_pub{0, 0, RCL_ROS_TIME};
    };

    static std::string guessEncoding(const cv::Mat& m) {
        if (m.type() == CV_8UC1) return "mono8";
        if (m.type() == CV_8UC3) return "bgr8";
        return m.channels() == 1 ? "mono8" : "bgr8";
    }

    std::string imageTopic(const std::string& tag) const { return image_ns_ + (tag.empty() ? "" : "/" + tag); }
    std::string cloudTopic(const std::string& tag) const { return cloud_ns_ + (tag.empty() ? "" : "/" + tag); }

    void loop() {
        using clock = std::chrono::steady_clock;
        auto next_wake = clock::now();
        while (running_) {
            next_wake += std::chrono::milliseconds(5);
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait_until(lk, next_wake, [&] { return !running_ || anyReady(); });
            if (!running_) break;

            for (auto& kv : image_channels_) {
                auto& ch = kv.second;
                if (!ch.initialized || !ch.has_data || !ch.pub) continue;
                if (ch.period > 0 && ch.last_pub.nanoseconds() != 0 &&
                    (node_->now() - ch.last_pub).seconds() < ch.period) {
                    continue;
                }
                std_msgs::msg::Header h;
                h.stamp = ch.stamp.nanoseconds() == 0 ? node_->now() : ch.stamp;
                auto msg = cv_bridge::CvImage(h, guessEncoding(ch.latest), ch.latest).toImageMsg();
                ch.pub->publish(*msg);
                ch.last_pub = node_->now();
                ch.has_data = false;
            }

            for (auto& kv : cloud_channels_) {
                auto& ch = kv.second;
                if (!ch.initialized || !ch.has_data || !ch.latest || !ch.pub) continue;
                if (ch.period > 0 && ch.last_pub.nanoseconds() != 0 &&
                    (node_->now() - ch.last_pub).seconds() < ch.period) {
                    continue;
                }
                sensor_msgs::msg::PointCloud2 msg;
                pcl::toROSMsg(*ch.latest, msg);
                msg.header.stamp = ch.stamp.nanoseconds() == 0 ? node_->now() : ch.stamp;
                msg.header.frame_id = ch.frame_id.empty() ? "map" : ch.frame_id;
                ch.pub->publish(msg);
                ch.last_pub = node_->now();
                ch.has_data = false;
            }
        }
    }

    bool anyReady() const {
        for (const auto& kv : image_channels_) if (kv.second.has_data) return true;
        for (const auto& kv : cloud_channels_) if (kv.second.has_data) return true;
        return false;
    }

private:
    rclcpp::Node::SharedPtr node_;
    std::string image_ns_, cloud_ns_;
    std::unordered_map<std::string, ImageCh> image_channels_;
    std::unordered_map<std::string, CloudCh> cloud_channels_;
    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
};

}  // namespace vloc
