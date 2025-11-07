#include "ros_localization_node.h"

namespace vloc {

RosLocalizationNode::RosLocalizationNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
: nh_(nh), pnh_(pnh), it_(nh_),
  left_sub_(nh_, "", 1), right_sub_(nh_, "", 1),
  sync_(SyncPolicy(10), left_sub_, right_sub_)
{
    // ROS 参数
    std::string left_topic = "/camera/left/image";
    std::string right_topic = "/camera/right/image";
    LocalizationCore::Params params; // 算法参数

    pnh_.param<std::string>("left_topic", left_topic, left_topic);
    pnh_.param<std::string>("right_topic", right_topic, right_topic);
    pnh_.param<std::string>("map_path", params.map_path, params.map_path);
    pnh_.param<std::string>("camera_yaml", params.camera_yaml, params.camera_yaml);
    pnh_.param<std::string>("superpoint_engine", params.superpoint_engine, params.superpoint_engine);
    pnh_.param<int>("image_width", params.image_width, 640);
    pnh_.param<int>("image_height", params.image_height, 400);
    pnh_.param<bool>("force_resize", params.force_resize, false);
    pnh_.param<bool>("use_clahe", params.use_clahe, false);
    pnh_.param<bool>("enable_debug", params.enable_debug, false);
    pnh_.param<std::string>("world_frame", world_frame_, world_frame_);
    pnh_.param<std::string>("camera_frame", camera_frame_, camera_frame_);
    pnh_.param<bool>("show_debug", show_debug_, false);

    // 统一发布线程参数
    pnh_.param<std::string>("debug_image_ns", debug_image_ns_, std::string("debug_image"));
    pnh_.param<std::string>("debug_cloud_ns", debug_cloud_ns_, std::string("debug_cloud"));
    pnh_.param<double>("debug_image_fps", debug_image_fps_, 30.0);
    pnh_.param<double>("debug_cloud_fps", debug_cloud_fps_, 5.0);

    // 初始化算法层
    if (!init(params)) {
        ROS_FATAL("LocalizationCore init failed");
        throw std::runtime_error("core init failed");
    }

    // 订阅与同步（stereoCallback 仅收图并入队）
    left_sub_.subscribe(nh_, left_topic, 1);
    right_sub_.subscribe(nh_, right_topic, 1);
    sync_.registerCallback(boost::bind(&RosLocalizationNode::stereoCallback, this, _1, _2));

    // 发布者
    pose_pub_ = pnh_.advertise<geometry_msgs::PoseStamped>("pose", 10);
    path_pub_ = pnh_.advertise<nav_msgs::Path>("path", 10);
    path_.header.frame_id = world_frame_;

    // 统一发布线程：注册图像与两路点云
    if (show_debug_) {
        debug_.reset(new DebugPublisherWorker(pnh_, debug_image_ns_, debug_cloud_ns_));
        debug_->addImageChannel("reproj", debug_image_fps_);      // ~debug_image
        debug_->addCloudChannel("map",   debug_cloud_fps_, "map");   // ~debug_cloud/map
        debug_->addCloudChannel("frame", debug_cloud_fps_, "map");   // ~debug_cloud/frame
        debug_->start();
    }

    // 启动计算线程
    start();
}
    RosLocalizationNode::~RosLocalizationNode() {
    if (debug_) debug_->stop();
}
void RosLocalizationNode::stereoCallback(const sensor_msgs::ImageConstPtr& left_msg,
                                         const sensor_msgs::ImageConstPtr& right_msg)
{
    // 只做最小处理：解码到灰度并入队
    cv::Mat left, right;
    try {
        auto lcv = cv_bridge::toCvShare(left_msg, left_msg->encoding);
        auto rcv = cv_bridge::toCvShare(right_msg, right_msg->encoding);
        left = lcv->image; right = rcv->image;
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    if (left.channels() == 3) cv::cvtColor(left, left, cv::COLOR_BGR2GRAY);
    if (right.channels() == 3) cv::cvtColor(right, right, cv::COLOR_BGR2GRAY);

    const double ts = left_msg->header.stamp.toSec();
    pushFrame(left, right, ts);
}

void RosLocalizationNode::onPose(const Eigen::Matrix4d& T_w_c, double timestamp_sec)
{
    ros::Time stamp; stamp.fromSec(timestamp_sec);

    geometry_msgs::PoseStamped ps; ps.header.stamp = stamp; ps.header.frame_id = world_frame_;
    Eigen::Quaterniond q(T_w_c.block<3,3>(0,0));
    ps.pose.position.x = T_w_c(0,3);
    ps.pose.position.y = T_w_c(1,3);
    ps.pose.position.z = T_w_c(2,3);
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    ps.pose.orientation.w = q.w();
    pose_pub_.publish(ps);

    path_.header.stamp = stamp;
    path_.poses.push_back(ps);
    path_pub_.publish(path_);

    static tf::TransformBroadcaster br;
    tf::Transform tfT;
    tf::Matrix3x3 R(
        T_w_c(0,0), T_w_c(0,1), T_w_c(0,2),
        T_w_c(1,0), T_w_c(1,1), T_w_c(1,2),
        T_w_c(2,0), T_w_c(2,1), T_w_c(2,2));
    tfT.setOrigin(tf::Vector3(T_w_c(0,3), T_w_c(1,3), T_w_c(2,3)));
    tf::Quaternion tq; R.getRotation(tq);
    tfT.setRotation(tq);
    br.sendTransform(tf::StampedTransform(tfT, stamp, world_frame_, camera_frame_));
}

void RosLocalizationNode::onDebugImage(const cv::Mat& img, double timestamp_sec)
{
    if (!show_debug_) return;
    cv::imshow("vloc_debug", img);
    cv::waitKey(1);
}

    void RosLocalizationNode::onDebugImage(const std::string& tag,
                                       const cv::Mat& img, double ts) {
    // if (!show_debug_) return;
    // auto& pub = dbg_pubs_[tag];
    // if (!pub) {
    //     const std::string topic = debug_ns_ + "/" + tag;   // e.g. ~debug/matches
    //     pub.reset(new ImagePublisherWorker(pnh_, topic, 1, debug_image_fps_));
    //     pub->start();
    // }
    // ros::Time stamp; stamp.fromSec(ts);
    // pub->post(img, stamp); // 异步发布
    if (!show_debug_ || !debug_) return;
    ros::Time t; t.fromSec(ts);
    debug_->postImage(tag, img, t);  // ~debug_image
}
    void RosLocalizationNode::onPointCloud(const std::string& tag,
                                       pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                       double timestamp_sec,
                                       const std::string& frame_id)
{
    if (!debug_) return;
    ros::Time t; t.fromSec(timestamp_sec);
    debug_->postCloud(tag, cloud, t, frame_id); // ~debug_cloud/<tag>
}

} // namespace vloc