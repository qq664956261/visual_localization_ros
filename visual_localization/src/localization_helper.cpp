#include "localization_helper.h"
#include <algorithm>


using namespace VISUAL_MAPPING;


namespace vloc {


    std::vector<int> LocalizationHelper::sort_frames_by_distance(std::vector<std::shared_ptr<Frame>> &frames, Eigen::Matrix3d R, Eigen::Vector3d t) {
        std::vector<int> sorted_ids;
        std::vector<double> sorted_distances;
        sorted_ids.reserve(frames.size());
        sorted_distances.reserve(frames.size());
        for (const auto& frame : frames) {
            Eigen::Vector3d t_ = frame->get_t();
            Eigen::Vector3d t_diff = t_ - t;
            double distance = t_diff.norm();
            sorted_distances.push_back(distance);
            sorted_ids.push_back(frame->id);
        }
        std::sort(sorted_ids.begin(), sorted_ids.end(), [&sorted_distances](int i, int j) {
            return sorted_distances[i] < sorted_distances[j];
        });
        return sorted_ids;
    }


    std::vector<int> LocalizationHelper::sort_frames_by_distance(std::vector<std::shared_ptr<Frame>>& frames, Eigen::Matrix4d T) {
        return sort_frames_by_distance(frames, T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
    }


} // namespace vloc