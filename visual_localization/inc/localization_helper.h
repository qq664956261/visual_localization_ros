#include <vector>
#include <memory>
#include <Eigen/Core>
#include "mapping.h" // Frame




namespace vloc {


struct LocalizationHelper {
static std::vector<int> sort_frames_by_distance(
 std::vector<std::shared_ptr<VISUAL_MAPPING::Frame>>& frames,
 Eigen::Matrix3d R,  Eigen::Vector3d t);


static std::vector<int> sort_frames_by_distance(
 std::vector<std::shared_ptr<VISUAL_MAPPING::Frame>>& frames,
 Eigen::Matrix4d T);
};


} // namespace vloc
