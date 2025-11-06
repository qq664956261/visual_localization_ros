#pragma once
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>

#include "localization_core.h"

namespace vloc {

class RosLocalizationNode : public LocalizationCore {
public:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;

    RosLocalizationNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~RosLocalizationNode() override = default;

protected:
    // 算法回调：在这里发布 ROS 消息
    void onPose(const Eigen::Matrix4d& T_w_c, double timestamp_sec) override;
    void onDebugImage(const cv::Mat& img, double timestamp_sec) override;

private:
    void stereoCallback(const sensor_msgs::ImageConstPtr& left_msg,
                        const sensor_msgs::ImageConstPtr& right_msg);

private:
    ros::NodeHandle nh_, pnh_;
    image_transport::ImageTransport it_;
    message_filters::Subscriber<sensor_msgs::Image> left_sub_, right_sub_;
    message_filters::Synchronizer<SyncPolicy> sync_;

    ros::Publisher pose_pub_;
    ros::Publisher path_pub_;
    nav_msgs::Path path_;
    std::string world_frame_ = "map";
    std::string camera_frame_ = "camera";
    bool show_debug_ = false;
};

} // namespace vloc






