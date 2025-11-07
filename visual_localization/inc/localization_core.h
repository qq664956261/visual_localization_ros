#pragma once
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>
#include <string>


// 你的核心库头文件
#include "map_save.h"
#include "visualization.h"
#include "mapping.h"
#include "matcher.h"
#include "bundle_adjustment.h"
#include "camera.h"
#include "localization_helper.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


namespace vloc {


    // 与 ROS 无关的算法层
    class LocalizationCore {
    public:
        struct Params {
            std::string map_path;
            std::string camera_yaml;
            std::string superpoint_engine;
            int image_width = 640;
            int image_height = 400;
            bool force_resize = false;
            bool use_clahe = false;
            bool enable_debug = false;
            // 可选：初始位姿、可视化等
        };


        LocalizationCore();
        virtual ~LocalizationCore();


        // 初始化（加载地图、相机、特征引擎、可视化线程）
        bool init(const Params& p);


        // 启停计算线程
        void start();
        void stop();


        // 推入一帧双目数据（只入队，不计算）
        void pushFrame(const cv::Mat& left_gray, const cv::Mat& right_gray, double timestamp_sec);


    protected:
        // 供子类覆写：位姿输出、调试图像输出
        virtual void onPose(const Eigen::Matrix4d& T_w_c, double timestamp_sec) {}
        virtual void onDebugImage(const cv::Mat& img, double timestamp_sec) {}
        virtual void onDebugImage(const std::string& tag, const cv::Mat& img, double ts)
        {
            // // 默认回落到单路接口，保持兼容
            // onDebugImage(img, ts);
        }
        virtual void onPointCloud(const std::string& tag,
            pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
            double timestamp_sec,
            const std::string& frame_id){}



        // 读取相机 yaml（含 T_c1_c2）
        static bool read_cam_params(const std::string& path, VISUAL_MAPPING::Camera& cam1,
        VISUAL_MAPPING::Camera& cam2, Eigen::Matrix4d& T12);


    private:
        struct StereoItem {
            cv::Mat left, right; // 灰度
            double ts = 0.0;
        };


        void computeLoop();
        bool popItem(StereoItem& out);


    private:
        // 参数
        Params params_;


        // 地图/相机/匹配/优化
        VISUAL_MAPPING::MapSaver map_saver_;
        VISUAL_MAPPING::Map map_;
        VISUAL_MAPPING::Matcher matcher_;
        VISUAL_MAPPING::Visualization vis_;
        std::thread vis_thread_;


        std::vector<std::shared_ptr<VISUAL_MAPPING::Frame>> frames_;
        VISUAL_MAPPING::Camera cam1_, cam2_;
        Eigen::Matrix4d T12_ = Eigen::Matrix4d::Identity();
        std::shared_ptr<VISUAL_MAPPING::FeatureDetection> detection_;


        // 状态
        Eigen::Matrix4d init_T_ = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d last_T_ = Eigen::Matrix4d::Identity();
        int frame_cnt_ = 0;

        // 新增：地图点云缓存 + 节流
        pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
        double last_map_pub_sec_ = 0.0;


        // 线程与队列
        std::thread worker_;
        std::mutex mtx_;
        std::condition_variable cv_;
        std::deque<StereoItem> q_;
        std::atomic<bool> running_{false};
        cv::Ptr<cv::CLAHE> clahe_;
    };


} // namespace vloc