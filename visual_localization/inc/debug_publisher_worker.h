#pragma once
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
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

        explicit DebugPublisherWorker(ros::NodeHandle& pnh,
                                      const std::string& image_ns = "debug_image",
                                      const std::string& cloud_ns = "debug_cloud")
        : pnh_(pnh), it_(pnh_), image_ns_(image_ns), cloud_ns_(cloud_ns) {}

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
        } // 添加图像/点云通道（懒初始化）
        void addImageChannel(const std::string& tag, double fps = 30.0) {
            std::lock_guard<std::mutex> lk(mtx_);
            auto& ch = image_channels_[tag];
            if (!ch.initialized) {
                ch.topic = imageTopic(tag);
                ch.pub = it_.advertise(ch.topic, 1);
                ch.fps = fps; ch.period = fps > 0 ? (1.0 / fps) : 0.0;
                ch.initialized = true;
            } else {
                ch.fps = fps; ch.period = fps > 0 ? 1.0 / fps : 0.0;
            }
        }
        void addCloudChannel(const std::string& tag, double fps = 5.0, const std::string& frame_id = "map") {
            std::lock_guard<std::mutex> lk(mtx_);
            auto& ch = cloud_channels_[tag];
            if (!ch.initialized) {
                ch.topic = cloudTopic(tag);
                ch.pub = pnh_.advertise<sensor_msgs::PointCloud2>(ch.topic, 1);
                ch.fps = fps; ch.period = fps > 0 ? (1.0 / fps) : 0.0;
                ch.frame_id = frame_id;
                ch.initialized = true;
            } else {
                ch.fps = fps; ch.period = fps > 0 ? 1.0 / fps : 0.0; ch.frame_id = frame_id;
            }
        }  // 投递最新数据（浅拷贝；若上游还要改同一 Mat，请传 clone）
        void postImage(const std::string& tag, const cv::Mat& img, const ros::Time& stamp) {
            addImageChannel(tag);
            std::lock_guard<std::mutex> lk(mtx_);
            auto& ch = image_channels_[tag];
            ch.latest = img.clone(); ch.stamp = stamp; ch.has_data = true;
            cv_.notify_one();
        }
        void postCloud(const std::string& tag, const CloudConstPtr& cloud, const ros::Time& stamp, const std::string& frame_id = "") {

            addCloudChannel(tag);
            std::lock_guard<std::mutex> lk(mtx_);
            auto& ch = cloud_channels_[tag];

            ch.latest = cloud; ch.stamp = stamp; if (!frame_id.empty()) ch.frame_id = frame_id; ch.has_data = true;
            cv_.notify_one();
        }

        void setImageNs(const std::string& ns) { image_ns_ = ns; }
        void setCloudNs(const std::string& ns) { cloud_ns_ = ns; }private:
        struct ImageCh {
            image_transport::Publisher pub; std::string topic; bool initialized = false;
            cv::Mat latest; ros::Time stamp; bool has_data = false;
            double fps = 30.0; double period = 1.0 / 30.0; ros::Time last_pub;
        };
        struct CloudCh {
            ros::Publisher pub; std::string topic; bool initialized = false;
            CloudConstPtr latest; ros::Time stamp; bool has_data = false;
            std::string frame_id = "map";
            double fps = 5.0; double period = 1.0 / 5.0; ros::Time last_pub;
        };

        static std::string guessEncoding(const cv::Mat& m) {
            if (m.type() == CV_8UC1) return "mono8";
            if (m.type() == CV_8UC3) return "bgr8";
            return m.channels() == 1 ? "mono8" : "bgr8";
        }
        std::string imageTopic(const std::string& tag) const {
            return image_ns_ + (tag.empty() ? "" : "/" + tag);
        }
        std::string cloudTopic(const std::string& tag) const {
            return cloud_ns_ + (tag.empty() ? "" : "/" + tag);
        }
 void loop() {
        using clock = std::chrono::steady_clock;
        auto next_wake = clock::now();
        while (running_) {
            next_wake += std::chrono::milliseconds(5);
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_until(lk, next_wake, [&]{ return !running_ || anyReady(); });
                if (!running_) break;

                // 发布图像
                for (auto& kv : image_channels_) {
                    auto& ch = kv.second;
                    if (!ch.initialized || !ch.has_data) continue;
                    if (ch.period > 0 && !ch.last_pub.isZero()) {
                        if ((ros::Time::now() - ch.last_pub).toSec() < ch.period) continue;
                    }
                    if (ch.pub) {
                        std_msgs::Header h; h.stamp = ch.stamp.isZero() ? ros::Time::now() : ch.stamp;
                        auto enc = guessEncoding(ch.latest);
                        auto msg = cv_bridge::CvImage(h, enc, ch.latest).toImageMsg();
                        ch.pub.publish(msg);
                        ch.last_pub = ros::Time::now();
                        ch.has_data = false;
                    }
                }

                // 发布点云
                for (auto& kv : cloud_channels_) {
                    auto& ch = kv.second;
                    if (!ch.initialized || !ch.has_data || !ch.latest) continue;
                    if (ch.period > 0 && !ch.last_pub.isZero()) {
                        if ((ros::Time::now() - ch.last_pub).toSec() < ch.period) continue;
                    }
                    if (ch.pub) {
                        sensor_msgs::PointCloud2 msg;
                        pcl::toROSMsg(*ch.latest, msg);
                        msg.header.stamp = ch.stamp.isZero() ? ros::Time::now() : ch.stamp;
                        msg.header.frame_id = ch.frame_id.empty() ? "map" : ch.frame_id;
                        ch.pub.publish(msg);
                        ch.last_pub = ros::Time::now();
                        ch.has_data = false;
                    }
                }
            }
        }
    }
        bool anyReady() const {
            for (auto& kv : image_channels_) if (kv.second.has_data) return true;
            for (auto& kv : cloud_channels_) if (kv.second.has_data) return true;
            return false;
        }

    private:
        ros::NodeHandle pnh_;
        image_transport::ImageTransport it_;
        std::string image_ns_, cloud_ns_;

        std::unordered_map<std::string, ImageCh> image_channels_;
        std::unordered_map<std::string, CloudCh> cloud_channels_;

        std::thread worker_;
        std::mutex mtx_;
        std::condition_variable cv_;
        std::atomic<bool> running_{false};
    };

} // namespace vloc
