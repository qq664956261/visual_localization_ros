#pragma once

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
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
        ImagePublisherWorker(ros::NodeHandle& nh,
        const std::string& topic = "debug_image",
        int queue_size = 1,
        double max_fps = 30.0)
        : nh_(nh), it_(nh_), topic_(topic), queue_(queue_size),
        target_period_(max_fps > 0 ? 1.0 / max_fps : 0.0) {}


        void start() {
            if (running_.exchange(true)) return;
            pub_ = it_.advertise(topic_, queue_);
            worker_ = std::thread(&ImagePublisherWorker::loop, this);
        }


        void stop() {
            if (!running_.exchange(false)) return;
            cv_.notify_all();
            if (worker_.joinable()) worker_.join();
            pub_.shutdown();
        }


        // 投递一帧（浅拷贝足够；如需隔离可改 clone()）
        void post(const cv::Mat& img, const ros::Time& stamp = ros::Time(0)) {
            std::lock_guard<std::mutex> lk(mtx_);
            latest_ = img.clone(); // 浅拷贝
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
    if (m.channels() == 1) return "mono8";
    if (m.channels() == 3) return "bgr8";
    return "bgr8"; // 兜底
}


void loop() {
    using clock = std::chrono::steady_clock;
    auto next_tick = clock::now();
    while (running_) {
        if (target_period_ > 0) {
            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(target_period_)
            );
        }


        cv::Mat frame; ros::Time t;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait_for(lk, std::chrono::milliseconds(5), [&]{ return !running_ || has_frame_; });
            if (!running_) break;
            if (has_frame_) {
                frame = latest_;
                t = stamp_;
                has_frame_ = false;
            }
        }
        if (!frame.empty() && pub_) {
            std_msgs::Header h; h.stamp = t.isZero() ? ros::Time::now() : t;
            auto enc = guessEncoding(frame);
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(h, enc, frame).toImageMsg();
            pub_.publish(msg);
        }


        if (target_period_ > 0) std::this_thread::sleep_until(next_tick);
    }
}


private:
ros::NodeHandle nh_;
image_transport::ImageTransport it_;
image_transport::Publisher pub_;
std::string topic_;
int queue_ = 1;


std::thread worker_;
std::mutex mtx_;
std::condition_variable cv_;
cv::Mat latest_;
ros::Time stamp_;
bool has_frame_ = false;
std::atomic<bool> running_{false};
double target_period_ = 0.0; // seconds
};
} // namespace vloc