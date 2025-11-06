#include <ros/ros.h>
#include "ros_localization_node.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "visual_localization_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    try {
        vloc::RosLocalizationNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("Exception: %s", e.what());
        return 1;
    }
    return 0;
}