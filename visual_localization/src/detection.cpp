#include "detection.h"
#include "opencv2/features2d.hpp"
#include "extractor.h"
#include "interface.h"

namespace VISUAL_MAPPING {

    FeatureDetection::FeatureDetection(int type, const std::string& weights_path, int nms, int num_kps, int width, int height) {
        this->type = type;
        this->weights_path = weights_path;
        this->nms_size = nms;
        this->num_kps = num_kps;
        this->IMG_WIDTH = width;
        this->IMG_HEIGHT = height;
        if (type == ALIKE && (width % 32 != 0 || height % 32 != 0)) {
            this->IMG_WIDTH = width % 32 != 0 ? width + 32 - width % 32 : width;
            this->IMG_HEIGHT = height % 32 != 0 ? height + 32 - height % 32 : height;
        }
        std::cout<<"set image size: "<<IMG_WIDTH<<"x"<<IMG_HEIGHT<<std::endl;
        if (type == ALIKE) {
            net_ptr = std::make_shared<Interface>("alike", weights_path + alike_model_path, true, cv::Size(IMG_WIDTH,IMG_HEIGHT));
            descriptor_dim = 64;
            descriptor_width = IMG_WIDTH;
            descriptor_height = IMG_HEIGHT;
        } else if (type == D2Net) {
            net_ptr = std::make_shared<Interface>("d2net", weights_path + d2net_model_path, true, cv::Size(IMG_WIDTH,IMG_HEIGHT));
            descriptor_dim = 512;
            descriptor_width = IMG_WIDTH / 8;
            descriptor_height = IMG_HEIGHT / 8;
        } else if (type == SuperPoint) {
            if (use_tensorrt)
            {
                net_tensorRT_ptr = std::make_shared<InterfaceTensorRT>("SuperPoint", weights_path, true,
                                              cv::Size(IMG_WIDTH,IMG_HEIGHT),
                                              IMG_WIDTH / 8, IMG_HEIGHT / 8, 256, true);
            }
            else
            {
                net_ptr = std::make_shared<Interface>("SuperPoint", weights_path + sp_model_path, true, cv::Size(IMG_WIDTH,IMG_HEIGHT), true);
            }
            descriptor_dim = 256;
            descriptor_width = IMG_WIDTH / 8;
            descriptor_height = IMG_HEIGHT / 8;
        } else if (type == DISK) {
            net_ptr = std::make_shared<Interface>("disk", weights_path + disk_model_path, true, cv::Size(IMG_WIDTH,IMG_HEIGHT));
            descriptor_dim = 128;
            descriptor_width = IMG_WIDTH;
            descriptor_height = IMG_HEIGHT;
        } else if (type == XFeat) {
            net_ptr = std::make_shared<Interface>("xfeat", weights_path + xfeat_model_path, true, cv::Size(IMG_WIDTH,IMG_HEIGHT));
            descriptor_dim = 64;
            descriptor_width = IMG_WIDTH / 8;
            descriptor_height = IMG_HEIGHT / 8;
        }
        score_map = cv::Mat(IMG_HEIGHT, IMG_WIDTH, CV_32FC1);
        desc_map = cv::Mat(descriptor_height, descriptor_width, CV_32FC(descriptor_dim));
    }

    void FeatureDetection::detectFeatures(cv::Mat &image, std::vector<Eigen::Vector2d> &features_uv, cv::Mat &descriptors) {
        cv::Mat mask = cv::Mat::ones(image.size(), CV_8UC1) * 255;
        // 最下方100行不检测特征点
        mask(cv::Rect(0, image.rows - 100, image.cols, 100)) = 0;

        if (type == SIFT) {
            // // 1. detect sift features
            // auto sift = cv::SIFT::create();
            // std::vector<cv::KeyPoint> keypoints;
            // sift->detect(image, keypoints);
            // for (auto &keypoint : keypoints) {
            //     features_uv.emplace_back(keypoint.pt.x, keypoint.pt.y);
            // }
            // // 2. compute descriptors
            // sift->compute(image, keypoints, descriptors);
        } else if (type == ORB) {
            // 1. detect orb features
            auto orb = cv::ORB::create();
            std::vector<cv::KeyPoint> keypoints;
            orb->detect(image, keypoints);
            for (auto &keypoint : keypoints) {
                features_uv.emplace_back(keypoint.pt.x, keypoint.pt.y);
            }
            // 2. compute descriptors
            orb->compute(image, keypoints, descriptors);
        } else if (type == SuperPoint || type == ALIKE || type == D2Net || type == DISK || type == XFeat) {
            int cols = image.cols;
            int rows = image.rows;
            if (type == ALIKE && (cols % 32 != 0 || rows % 32 != 0)) {
                int new_cols = cols % 32 != 0 ? cols + 32 - cols % 32 : cols;
                int new_rows = rows % 32 != 0 ? rows + 32 - rows % 32 : rows;
                cv::Mat new_image(new_rows, new_cols, image.type(), cv::Scalar(0));
                image.copyTo(new_image(cv::Rect(0, 0, cols, rows)));
                net_ptr->run(new_image, score_map, desc_map);
                std::vector<cv::KeyPoint> key_points;

                cv::Mat new_mask = cv::Mat::ones(new_image.size(), CV_8UC1) * 255;
                // 最下方100行不检测特征点
                new_mask(cv::Rect(0, new_image.rows - 100, new_image.cols, 100)) = 0;
                key_points = nms(score_map, num_kps, 0.01, nms_size, new_mask);

                for (auto &keypoint : key_points) {
                    features_uv.emplace_back(keypoint.pt.x, keypoint.pt.y);
                }
                descriptors = bilinear_interpolation(new_image.cols, new_image.rows, desc_map, key_points);
            } else {
                if (use_tensorrt)
                {
                    //std::cout<<"image.type():"<<image.type()<<std::endl;
                    net_tensorRT_ptr->run(image, score_map, desc_map);
                }
                else
                {
                    net_ptr->run(image, score_map, desc_map);
                }

                std::vector<cv::KeyPoint> key_points;
                key_points = nms(score_map, num_kps, 0.01, nms_size, cv::Mat());


                // std::ofstream ofs("key_points.txt");
                // ofs << std::setprecision(15);
                // ofs << key_points.size() << "\n";
                // for (size_t i = 0; i < key_points.size(); ++i) {
                //     //std::cout<<""<<std::endl;
                //     ofs << i << " " << key_points[i].pt.x << " " << key_points[i].pt.y << "\n";
                // }
      


                for (auto &keypoint : key_points) {
                    features_uv.emplace_back(keypoint.pt.x, keypoint.pt.y);
                }
                descriptors = bilinear_interpolation(image.cols, image.rows, desc_map, key_points);

                // std::ofstream ofs1("descriptors.txt");
                // ofs1 << descriptors.rows << " " << descriptors.cols << "\n";
                // ofs1 << std::setprecision(15);
                // for (int r = 60; r < 70; ++r) {
                //     ofs1 << r;
                //     for (int c = 0; c < descriptors.cols; ++c) {
                //         float v = descriptors.at<float>(r, c);
                //         ofs1 << " " << v;
                //     }
                //     ofs1 << "\n";
                // }





                
    // cv::Mat image1;

    // image1 = image.clone();

    // // 把匹配的点画上去并连线
    // for (size_t i = 0; i < features_uv.size(); ++i) {

    //     // 点坐标
    //     cv::Point2f pt_l(features_uv[i].x(),  features_uv[i].y());
    //     // 颜色随便来一个
    //     cv::Scalar color(
    //         (unsigned) (37 * i % 255),
    //         (unsigned) (17 * i % 255),
    //         (unsigned) (97 * i % 255)
    //     );
    //     // 画点
    //     cv::circle(image1, pt_l, 3, color, -1, cv::LINE_AA);
    // }

    // // 4. 显示 / 保存
    // cv::imshow("featurepoints", image1);
    // cv::waitKey(1); // 调试时也可以用 0 暂停
            }
        } else {
            std::cout<<"model type not supported"<<std::endl;
        }
    }

} // namespace reusable_map
