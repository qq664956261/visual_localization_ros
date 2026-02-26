#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include "localization_core.h"
#include "image_publisher_worker.h"
#include "debug_publisher_worker.h"

namespace vloc {

class RosLocalizationNode : public LocalizationCore {
public:
    using ImageMsg = sensor_msgs::msg::Image;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;

    explicit RosLocalizationNode(const rclcpp::Node::SharedPtr& node);
    ~RosLocalizationNode() override;

protected:
    void onPose(const Eigen::Matrix4d& T_w_c, double timestamp_sec) override;
    void onDebugImage(const cv::Mat& img, double timestamp_sec) override;
    void onDebugImage(const std::string& tag, const cv::Mat& img, double ts) override;
    void onPointCloud(const std::string& tag,
                      pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                      double timestamp_sec,
                      const std::string& frame_id) override;

private:
    void stereoCallback(const ImageMsg::ConstSharedPtr& left_msg,
                        const ImageMsg::ConstSharedPtr& right_msg);

private:
    rclcpp::Node::SharedPtr node_;
    message_filters::Subscriber<ImageMsg> left_sub_, right_sub_;
    message_filters::Synchronizer<SyncPolicy> sync_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    nav_msgs::msg::Path path_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string world_frame_ = "map";
    std::string camera_frame_ = "camera";
    bool show_debug_ = false;

    std::unique_ptr<DebugPublisherWorker> debug_;
    std::string debug_image_ns_ = "debug_image";
    std::string debug_cloud_ns_ = "debug_cloud";
    double debug_image_fps_ = 30.0;
    double debug_cloud_fps_ = 5.0;
};

} // namespace vloc
