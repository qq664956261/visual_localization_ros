#include <rclcpp/rclcpp.hpp>
#include "ros_localization_node.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("visual_localization_node");

    try {
        auto app = std::make_shared<vloc::RosLocalizationNode>(node);
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(node->get_logger(), "Exception: %s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}
