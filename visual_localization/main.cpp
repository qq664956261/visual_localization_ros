#include <iostream>
#include <thread>
#include "frame.h"
#include "mapping.h"
#include "fstream"
#include "triangulation.h"
#include "bundle_adjustment.h"
#include "visualization.h"
#include "matcher.h"
#include "detection.h"
#include "camera.h"
#include "map_save.h"

// read yaml opencv
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/opencv.hpp"

using namespace VISUAL_MAPPING;

std::vector<std::pair<std::string, std::string>> read_img_path(const std::string& path1, const std::string& path2, const std::string& list, std::vector<Eigen::Matrix4d>& T){
    std::fstream pose;
    pose.open("pose.txt", std::ios::out);
    std::ifstream file;
    file.open(list);
    bool first_pose = true;
    Eigen::Matrix4d Tw1;
    std::vector<std::pair<std::string, std::string>> img_list;
    std::string line;
    while (std::getline(file, line, ' ')) {
        img_list.emplace_back(path1 + line + ".png", path2 + line + ".png");
        std::string time = line;
        Eigen::Matrix4d T_;
        Eigen::Vector3d t;
        Eigen::Quaterniond q;
        for (int i = 0;i < 6;i ++) {
            std::getline(file, line, ' ');
            if (i == 0) {
                t(0) = std::stod(line);
            } else if (i == 1) {
                t(1) = std::stod(line);
            } else if (i == 2) {
                t(2) = std::stod(line);
            } else if (i == 3) {
                q.x() = std::stod(line);
            } else if (i == 4) {
                q.y() = std::stod(line);
            } else if (i == 5) {
                q.z() = std::stod(line);
            }
        }
        std::getline(file, line, '\n');
        q.w() = std::stod(line);
        T_.block<3,3>(0,0) = q.toRotationMatrix();
        T_.block<3,1>(0,3) = t;
        T_.row(3) << 0, 0, 0, 1;
        if(first_pose)
        {
            first_pose = false;
            Tw1 = T_;
        }
        T_ = Tw1.inverse() * T_;
        Eigen::Matrix4d T_cam_lidar;
        //std::cout<<"1T_:"<<T_<<std::endl;
        
        T_cam_lidar.setIdentity();
        //z轴90 * y轴-90
        T_cam_lidar.row(0) << 0, -1, 0, 0;
        T_cam_lidar.row(1) << 0, 0, -1, 0;
        T_cam_lidar.row(2) << 1, 0, 0, 0;
        //std::cout<<"T_cam_lidar:"<<T_cam_lidar<<std::endl;

        
        //什么时候要用“夹心”B * T * B-1,当你处理的是同一段相对运动，只是想把它换个轴系来表达（比如“把 LiDAR 里的相对位姿用相机轴系来表示”），这才是被动换基
        T_ = T_cam_lidar * T_  * T_cam_lidar.inverse();
        //std::cout<<"2T_:"<<T_<<std::endl;
        Eigen::Quaterniond q_out(T_.block<3,3>(0,0));
        Eigen::Vector3d t_out;
        t_out = T_.block<3,1>(0,3);
        
        pose << time << " "<<t_out[0]<< " " <<  t_out[1] << " " << t_out[2] << " " << q_out.x() << " " << q_out.y()<< " "<< q_out.z() << " "<< q_out.w() << std::endl;
        T.push_back(T_); 
    }
    return img_list;
}

void read_cam_params(const std::string& path, Camera& cam1, Camera& cam2, Eigen::Matrix4d& T12) {
    // yaml file
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open " << path << std::endl;
        return;
    }
    std::string type = fs["Camera.type"];
    if (type == "pinhole") {
        double fx = fs["fx"];
    } else if (type == "KannalaBrandt8") {
        cam1.setModelType(KANNALA_BRANDT8);
        cam2.setModelType(KANNALA_BRANDT8);
        double fx = fs["Camera1.fx"];
        double fy = fs["Camera1.fy"];
        double cx = fs["Camera1.cx"];
        double cy = fs["Camera1.cy"];
        double k1 = fs["Camera1.k1"];
        double k2 = fs["Camera1.k2"];
        double k3 = fs["Camera1.k3"];
        double k4 = fs["Camera1.k4"];
        auto param2 = new Camera::KannalaBrandt8Params(fx, fy, cx, cy, k1, k2, k3, k4);
        cam1.setKannalaBrandt8Params(*param2);
        fx = fs["Camera2.fx"];
        fy = fs["Camera2.fy"];
        cx = fs["Camera2.cx"];
        cy = fs["Camera2.cy"];
        k1 = fs["Camera2.k1"];
        k2 = fs["Camera2.k2"];
        k3 = fs["Camera2.k3"];
        k4 = fs["Camera2.k4"];
        param2 = new Camera::KannalaBrandt8Params(fx, fy, cx, cy, k1, k2, k3, k4);
        cam2.setKannalaBrandt8Params(*param2);
    }
    cv::Mat T12_;
    fs["Stereo.T_c1_c2"] >> T12_;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float a = T12_.at<float>(i, j);
            T12(i, j) = a;
        }
    }
    fs.release();
}


int main() {
    // 1. read images list and camera parameters
    // std::string img_path = "/home/zc/data/recording_2020-12-22_12-04-35_stereo_images_undistorted/recording_2020-12-22_12-04-35/undistorted_images/cam0/";
    // std::string img_path2 = "/home/zc/data/recording_2020-12-22_12-04-35_stereo_images_undistorted/recording_2020-12-22_12-04-35/undistorted_images/cam1/";
    // std::string img_list_path = "/home/zc/code/class_4/visual_mapping_localization/kf_corridor2_02.txt";
    std::string img_path = "/home/zc/data/4seasons_export_pose/undistorted_images/cma0/";
    std::string img_path2 = "/home/zc/data/4seasons_export_pose/undistorted_images/cma1/";
    std::string img_list_path = "/home/zc/data/4seasons_export_pose/vehicle_pose.txt";
    std::vector<Eigen::Matrix4d> T;
    auto img_list = read_img_path(img_path, img_path2, img_list_path, T);

    // std::string cam_param_path = "/home/zc/code/ORB_SLAM3-master/Examples/Stereo/4seasons.yaml";
    std::string cam_param_path = "/home/zc/data/4seasons_export_pose/my.yaml";
    Camera cam1, cam2;
    Eigen::Matrix4d T12;
    read_cam_params(cam_param_path, cam1, cam2, T12);

    std::string feature_list_path = "/home/vio/Dataset/features.yaml";
    std::vector<std::string> feature_list;

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    // 2. create frame
//    img_list.resize(20);
    std::vector<std::shared_ptr<Frame>> frames;

    // std::shared_ptr<FeatureDetection> detection =
    //         std::make_shared<FeatureDetection>(SuperPoint, "/home/zc/code/learned_features_inference-openvino/weight/superpoint",
    //                                            8, 1000, 800, 400);
    std::shared_ptr<FeatureDetection> detection =
    std::make_shared<FeatureDetection>(SuperPoint, "/home/zc/code/learned_features_inference-openvino/weight/superpoint_640*400",
                                       16, 500, 640, 400);
    // img_list.erase(img_list.begin(), img_list.begin() + 157);
    // T.erase(T.begin(), T.begin() + 157);
    Eigen::Matrix4d init_T = *T.begin();
    for (auto & i : T) {
        i = init_T.inverse() * i;
    }
    for (int i = 0; i < img_list.size(); i ++) {
        cv::Mat img1 = cv::imread(img_list[i].first, cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(img_list[i].second, cv::IMREAD_GRAYSCALE);
        clahe->apply(img1, img1);
        clahe->apply(img2, img2);
        // cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
        // cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);


        auto frame = std::make_shared<Frame>(i, detection, T[i], img1, img2, &cam1, &cam2, T12);
        std::cout<<"frame "<<i<<std::endl;
        frames.push_back(frame);
//        cv::imshow("show", img1);
//        cv::waitKey(0);
    }

    // 3. init mapping and refine
    Mapping mapping;
    mapping.construct_initial_map(frames);
    for (int i = 0;i < 2; i++) {
        std::cout<<"start refine map "<<i<<std::endl;
        //mapping.refine_map();
    }

    // 4. save map
    MapSaver  mapSaver;
    mapSaver.save_map("map_sp.txt", frames, mapping.map);

//    MapSaver mapSaver2;
//    Mapping mapping2;
//    std::vector<std::shared_ptr<Frame>> frames2;
//    mapSaver2.load_map("map.txt", frames2, mapping2.map);

    // 5. visualization thread
    Visualization vis;
    std::thread vis_thread(Visualization::run, &vis, std::ref(mapping.map));
    vis_thread.join();

}
