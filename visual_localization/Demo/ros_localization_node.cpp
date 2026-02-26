#include "ros_localization_node.h"

namespace vloc {

RosLocalizationNode::RosLocalizationNode(const rclcpp::Node::SharedPtr& node)
    : node_(node),
      left_sub_(node_.get(), "", rmw_qos_profile_sensor_data),
      right_sub_(node_.get(), "", rmw_qos_profile_sensor_data),
      sync_(SyncPolicy(10), left_sub_, right_sub_) {
    std::string left_topic = "/camera/left/image";
    std::string right_topic = "/camera/right/image";
    LocalizationCore::Params params;

    node_->declare_parameter("left_topic", left_topic);
    node_->declare_parameter("right_topic", right_topic);
    node_->declare_parameter("map_path", params.map_path);
    node_->declare_parameter("camera_yaml", params.camera_yaml);
    node_->declare_parameter("superpoint_engine", params.superpoint_engine);
    node_->declare_parameter("image_width", 640);
    node_->declare_parameter("image_height", 400);
    node_->declare_parameter("force_resize", false);
    node_->declare_parameter("use_clahe", false);
    node_->declare_parameter("enable_debug", false);
    node_->declare_parameter("world_frame", world_frame_);
    node_->declare_parameter("camera_frame", camera_frame_);
    node_->declare_parameter("show_debug", false);
    node_->declare_parameter("debug_image_ns", std::string("debug_image"));
    node_->declare_parameter("debug_cloud_ns", std::string("debug_cloud"));
    node_->declare_parameter("debug_image_fps", 30.0);
    node_->declare_parameter("debug_cloud_fps", 5.0);

    node_->get_parameter("left_topic", left_topic);
    node_->get_parameter("right_topic", right_topic);
    node_->get_parameter("map_path", params.map_path);
    node_->get_parameter("camera_yaml", params.camera_yaml);
    node_->get_parameter("superpoint_engine", params.superpoint_engine);
    node_->get_parameter("image_width", params.image_width);
    node_->get_parameter("image_height", params.image_height);
    node_->get_parameter("force_resize", params.force_resize);
    node_->get_parameter("use_clahe", params.use_clahe);
    node_->get_parameter("enable_debug", params.enable_debug);
    node_->get_parameter("world_frame", world_frame_);
    node_->get_parameter("camera_frame", camera_frame_);
    node_->get_parameter("show_debug", show_debug_);
    node_->get_parameter("debug_image_ns", debug_image_ns_);
    node_->get_parameter("debug_cloud_ns", debug_cloud_ns_);
    node_->get_parameter("debug_image_fps", debug_image_fps_);
    node_->get_parameter("debug_cloud_fps", debug_cloud_fps_);

    if (!init(params)) {
        RCLCPP_FATAL(node_->get_logger(), "LocalizationCore init failed");
        throw std::runtime_error("core init failed");
    }

    left_sub_.subscribe(node_.get(), left_topic, rmw_qos_profile_sensor_data);
    right_sub_.subscribe(node_.get(), right_topic, rmw_qos_profile_sensor_data);
    sync_.registerCallback(std::bind(&RosLocalizationNode::stereoCallback, this, std::placeholders::_1, std::placeholders::_2));

    pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 10);
    path_pub_ = node_->create_publisher<nav_msgs::msg::Path>("path", 10);
    path_.header.frame_id = world_frame_;

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);

    if (show_debug_) {
        debug_ = std::make_unique<DebugPublisherWorker>(node_, debug_image_ns_, debug_cloud_ns_);
        debug_->addImageChannel("reproj", debug_image_fps_);
        debug_->addCloudChannel("map", debug_cloud_fps_, "map");
        debug_->addCloudChannel("frame", debug_cloud_fps_, "map");
        debug_->start();
    }

    start();
}

RosLocalizationNode::~RosLocalizationNode() {
    if (debug_) debug_->stop();
}

void RosLocalizationNode::stereoCallback(const ImageMsg::ConstSharedPtr& left_msg,
                                         const ImageMsg::ConstSharedPtr& right_msg) {
    cv::Mat left, right;
    try {
        auto lcv = cv_bridge::toCvShare(left_msg, left_msg->encoding);
        auto rcv = cv_bridge::toCvShare(right_msg, right_msg->encoding);
        left = lcv->image;
        right = rcv->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    if (left.channels() == 3) cv::cvtColor(left, left, cv::COLOR_BGR2GRAY);
    if (right.channels() == 3) cv::cvtColor(right, right, cv::COLOR_BGR2GRAY);

    const double ts = rclcpp::Time(left_msg->header.stamp).seconds();
    pushFrame(left, right, ts);
}

void RosLocalizationNode::onPose(const Eigen::Matrix4d& T_w_c, double timestamp_sec) {
    const auto stamp = rclcpp::Time::from_seconds(timestamp_sec);
    Eigen::Matrix4d T_w_l = T_lidar_cam * T_w_c * T_lidar_cam.inverse();

    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = stamp;
    ps.header.frame_id = world_frame_;
    Eigen::Quaterniond q(T_w_l.block<3, 3>(0, 0));
    ps.pose.position.x = T_w_l(0, 3);
    ps.pose.position.y = T_w_l(1, 3);
    ps.pose.position.z = T_w_l(2, 3);
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    ps.pose.orientation.w = q.w();
    pose_pub_->publish(ps);

    path_.header.stamp = stamp;
    path_.poses.push_back(ps);
    path_pub_->publish(path_);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = world_frame_;
    tf_msg.child_frame_id = camera_frame_;
    tf_msg.transform.translation.x = T_w_c(0, 3);
    tf_msg.transform.translation.y = T_w_c(1, 3);
    tf_msg.transform.translation.z = T_w_c(2, 3);

    Eigen::Quaterniond q_cam(T_w_c.block<3, 3>(0, 0));
    tf_msg.transform.rotation.x = q_cam.x();
    tf_msg.transform.rotation.y = q_cam.y();
    tf_msg.transform.rotation.z = q_cam.z();
    tf_msg.transform.rotation.w = q_cam.w();
    tf_broadcaster_->sendTransform(tf_msg);
}

void RosLocalizationNode::onDebugImage(const cv::Mat& img, double) {
    if (!show_debug_) return;
    cv::imshow("vloc_debug", img);
    cv::waitKey(1);
}

void RosLocalizationNode::onDebugImage(const std::string& tag, const cv::Mat& img, double ts) {
    if (!show_debug_ || !debug_) return;
    debug_->postImage(tag, img, rclcpp::Time::from_seconds(ts));
}

void RosLocalizationNode::onPointCloud(const std::string& tag,
                                       pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                       double timestamp_sec,
                                       const std::string& frame_id) {
    if (!debug_) return;
    debug_->postCloud(tag, cloud, rclcpp::Time::from_seconds(timestamp_sec), frame_id);
}

}  // namespace vloc
