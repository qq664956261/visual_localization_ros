#include "localization_core.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace VISUAL_MAPPING;

namespace vloc {

LocalizationCore::LocalizationCore() {}
LocalizationCore::~LocalizationCore() { stop(); }

bool LocalizationCore::init(const Params& p)
{
    params_ = p;

    // 加载地图
    if (params_.map_path.empty()) {
        std::cerr << "[LocalizationCore] map_path empty" << std::endl;
        return false;
    }
    map_saver_.load_map(params_.map_path, frames_, map_);

    // 读取相机
    if (params_.camera_yaml.empty() ||
        !read_cam_params(params_.camera_yaml, cam1_, cam2_, T12_)) {
        std::cerr << "[LocalizationCore] camera yaml failed" << std::endl;
        return false;
    }
    for (auto& f : frames_) f->camera = &cam1_;

    // 特征检测器
    if (params_.superpoint_engine.empty()) {
        std::cerr << "[LocalizationCore] superpoint_engine empty" << std::endl;
    }
    detection_ = std::make_shared<FeatureDetection>(SuperPoint, params_.superpoint_engine,
                                                    8, 1000, params_.image_width, params_.image_height);

    if (params_.use_clahe) clahe_ = cv::createCLAHE(3.0, cv::Size(8,8));

    // 可视化线程
    vis_thread_ = std::thread(Visualization::run, &vis_, std::ref(map_));

    init_T_.setIdentity();
    last_T_.setIdentity();
    frame_cnt_ = 0;

    return true;
}

void LocalizationCore::start()
{
    if (running_.exchange(true)) return;
    worker_ = std::thread(&LocalizationCore::computeLoop, this);
}

void LocalizationCore::stop()
{
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (vis_thread_.joinable()) vis_thread_.join();
}

void LocalizationCore::pushFrame(const cv::Mat& left_gray, const cv::Mat& right_gray, double timestamp_sec)
{
    StereoItem item;
    if (params_.force_resize) {
        cv::resize(left_gray, item.left, cv::Size(params_.image_width, params_.image_height));
        cv::resize(right_gray, item.right, cv::Size(params_.image_width, params_.image_height));
    } else {
        item.left = left_gray; item.right = right_gray;
    }
    if (params_.use_clahe && !item.left.empty() && !item.right.empty()) {
        clahe_->apply(item.left, item.left);
        clahe_->apply(item.right, item.right);
    }
    item.ts = timestamp_sec;

    {
        std::lock_guard<std::mutex> lk(mtx_);
        q_.push_back(std::move(item));
        while (q_.size()> 1)
            q_.pop_front();
    }
    cv_.notify_one();
}

bool LocalizationCore::popItem(StereoItem& out)
{
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [&]{ return !running_ || !q_.empty(); });
    if (!running_ && q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
}

void LocalizationCore::computeLoop()
{
    while (running_) {
        StereoItem item; if (!popItem(item)) break;
        if (item.left.empty() || item.right.empty()) continue;

        // 构造目标帧
        ++frame_cnt_;
        std::shared_ptr<Frame> tgt = std::make_shared<Frame>(frame_cnt_, detection_, init_T_,
                                                             item.left, item.right, &cam1_, &cam2_, T12_);
        if (frames_.empty()) continue;

        // 选最近关键帧
        auto close_ids = LocalizationHelper::sort_frames_by_distance(frames_, init_T_);

        // 与最近 3 帧做匹配与优化
        BundleAdjustment ba;
        for (int k = 0; k < std::min<int>(3, static_cast<int>(close_ids.size())); ++k) {
            auto ref = frames_[close_ids[k]];
            tgt->map_points.resize(tgt->map_points.size(), nullptr);

            // KNN 匹配
            auto matches = matcher_.match_knn(*ref, *tgt);
            int matches_num = 0;
            for (auto &m : matches) {
                if (ref->map_points[m.first] != nullptr) {
                    tgt->map_points[m.second] = ref->map_points[m.first];
                    matches_num++;
                }
            }
            if (matches_num > 30) {
                tgt->set_T(init_T_);
                auto inliers = ba.optimize_pose(tgt);
                for (size_t j = 0; j < inliers.size(); ++j) if (!inliers[j]) tgt->map_points[j] = nullptr;
            }

            // 重投影匹配
            matches = matcher_.match_re_projective(ref, tgt);
            matches_num = 0;
            for (auto &m : matches) {
                if (ref->map_points[m.first] != nullptr) {
                    tgt->map_points[m.second] = ref->map_points[m.first];
                    matches_num++;
                }
            }
            if (matches_num > 30) {
                tgt->set_T(init_T_);
                auto inliers = ba.optimize_pose(tgt);
                int inliers_num = 0;
                for (size_t j = 0; j < inliers.size(); ++j) {
                    if (!inliers[j]) tgt->map_points[j] = nullptr; else inliers_num++;
                }
                if (inliers_num > 30) break;
            }
        }

        // 连接帧补点
        auto connected = frames_[close_ids[0]]->get_connected_frames();
        for (auto & ref : connected) {
            auto matches = matcher_.match_projective(*tgt, ref->map_points);
            for (auto &m : matches) {
                if (tgt->map_points[m.first] == nullptr && ref->map_points[m.second] != nullptr) {
                    tgt->map_points[m.first] = ref->map_points[m.second];
                }
            }
        }

        // 最终一次 BA
        {
            BundleAdjustment ba2;
            auto inliers = ba2.optimize_pose(tgt);
            for (size_t j = 0; j < inliers.size(); ++j) if (!inliers[j]) tgt->map_points[j] = nullptr;
        }

        // 更新位姿 + 运动模型
        init_T_ = tgt->get_T();
        Eigen::Matrix4d dT = init_T_ * last_T_.inverse();
        last_T_ = init_T_;
        init_T_ = init_T_ * dT;

        // 可视化
        vis_.add_current_frame(tgt);

        // 调试图
        if (params_.enable_debug) {
            cv::Mat show = tgt->image.clone();
            if (show.channels() == 1) cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
            for (size_t i = 0; i < tgt->map_points.size(); ++i) {
                const auto& mp = tgt->map_points[i];
                if (!mp) continue;
                Eigen::Vector3d P = mp->x3D;
                Eigen::Vector3d Pc = tgt->get_R().transpose() * (P - tgt->get_t());
                Eigen::Vector2d uv = tgt->get_camera()->project(Pc);
                cv::circle(show, cv::Point((int)uv[0], (int)uv[1]), 3, cv::Scalar(0,255,0), 2);
                cv::circle(show, cv::Point((int)tgt->get_features_uv()[i][0], (int)tgt->get_features_uv()[i][1]), 2, cv::Scalar(0, 0, 255), 2);

            }
            onDebugImage(show, item.ts);
        }

        // 回调位姿
        onPose(init_T_, item.ts);
    }
}

bool LocalizationCore::read_cam_params(const std::string& path, Camera& cam1, Camera& cam2, Eigen::Matrix4d& T12)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    std::string type; if (fs["Camera.type"].isNone()) type = "KannalaBrandt8"; else fs["Camera.type"] >> type;
    auto load_kb = [&](const std::string& prefix, Camera& cam){
        double fx=0, fy=0, cx=0, cy=0, k1=0, k2=0, k3=0, k4=0;
        fs[prefix+".fx"] >> fx; fs[prefix+".fy"] >> fy;
        fs[prefix+".cx"] >> cx; fs[prefix+".cy"] >> cy;
        fs[prefix+".k1"] >> k1; fs[prefix+".k2"] >> k2;
        fs[prefix+".k3"] >> k3; fs[prefix+".k4"] >> k4;
        Camera::KannalaBrandt8Params p(fx, fy, cx, cy, k1, k2, k3, k4);
        cam.setKannalaBrandt8Params(p);
    };

    if (type == "KannalaBrandt8") {
        cam1.setModelType(KANNALA_BRANDT8);
        cam2.setModelType(KANNALA_BRANDT8);
        load_kb("Camera1", cam1);
        load_kb("Camera2", cam2);
    }

    cv::Mat T; if (!fs["Stereo.T_c1_c2"].isNone()) fs["Stereo.T_c1_c2"] >> T; else if (!fs["T_c1_c2"].isNone()) fs["T_c1_c2"] >> T; else return false;
    T12.setIdentity();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            T12(i,j) = T.depth()==CV_32F ? (double)T.at<float>(i,j) : T.at<double>(i,j);
    fs.release();
    return true;
}

} // namespace vloc

