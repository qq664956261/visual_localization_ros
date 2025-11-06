#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "matcher.h"

namespace VISUAL_MAPPING {
    void Matcher::match_bf(Frame *frame1, Frame *frame2,
                           std::vector<cv::DMatch> &matches) {

    }

    void Matcher::match_epipolar(Frame *frame1, Frame *frame2,
                                               std::vector<cv::DMatch> &matches) {

    }

    std::vector<std::pair<int, int>> Matcher::match_re_projective(std::shared_ptr<Frame> frame1, std::shared_ptr<Frame> frame2) {
        std::vector<std::pair<int, int>> matches;
        for (int i = 0;i < frame1->map_points.size(); i++) {
            if (frame1->map_points[i] != nullptr) {
                // 1. project the point to the second frame
                Eigen::Vector3d x3D = frame1->map_points[i]->x3D;
                Eigen::Vector3d x3D_2 = frame2->get_R().transpose() * (x3D - frame2->get_t());
                Eigen::Vector2d uv = frame2->get_camera()->project(x3D_2);
//                std::cout<<"features_uv: "<<frame1->get_features_uv()[i].transpose()<<std::endl;
//                std::cout << "uv: " << uv.transpose() << std::endl;
                // 2. get around features
                std::vector<int> around_ids = frame2->get_around_features(uv, 50);
                // 3. match the features
                cv::Mat desc_1 = frame1->descriptors.row(i);
                double min_dist = 1000;
                int min_id = -1;
                for (int around_id : around_ids) {
                    cv::Mat desc_2 = frame2->descriptors.row(around_id);
                    double dist = cv::norm(desc_1, desc_2);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_id = around_id;
                    }
                }
                if (min_id != -1 && min_dist < 1000) {
//                    if (frame2->features_depth[min_id] > 0)
                    {
                        // 4. check the reprojection error
//                        Eigen::Vector3d x3D_ = frame2->map_points[min_id]->x3D;
//                        Eigen::Vector3d x3D_1 = frame1->get_R().transpose() * (x3D_2 - frame1->get_t());
//                        Eigen::Vector2d uv_1 = frame1->get_camera()->project(x3D_1);
                        double error = (uv - frame2->get_features_uv()[min_id]).norm();
                        if (error < 15) {
                            matches.emplace_back(i, min_id);
                        }
                    }
                }

//                cv::Mat img1 = frame1->image.clone();
//                cv::Mat img2 = frame2->image.clone();
//                cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
//                cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);
//                cv::circle(img1, cv::Point((int)frame1->get_features_uv()[i].x(), (int)frame1->get_features_uv()[i].y()), 2, cv::Scalar(0, 255, 0), 2);
//
//                for (int i = 0; i < frame2->get_features_uv().size(); i++) {
//                    if (frame2->features_depth[i] > 0) {
//                        cv::circle(img2, cv::Point((int)frame2->get_features_uv()[i].x(), (int)frame2->get_features_uv()[i].y()), 3, cv::Scalar(0, 255, 0), 3);
//                    }
//                }
//                // around_ids
//                for (int around_id : around_ids) {
//                    if (frame2->features_depth[around_id] > 0)
//                    {
//                        cv::circle(img2, cv::Point((int) frame2->get_features_uv()[around_id].x(),
//                                                   (int) frame2->get_features_uv()[around_id].y()), 2,
//                                   cv::Scalar(0, 0, 255), 2);
//                    }
//                }
//                if (min_id != -1 && min_dist < 100) {
//                    cv::circle(img2, cv::Point((int)frame2->get_features_uv()[min_id].x(), (int)frame2->get_features_uv()[min_id].y()), 3, cv::Scalar(0, 0, 255), 3);
//                }
//                cv::circle(img2, cv::Point((int)uv.x(), (int)uv.y()), 1, cv::Scalar(255, 0, 0), 2);
//
//                cv::Mat show = cv::Mat(std::max(img1.rows, img2.rows), img1.cols + img2.cols, CV_8UC3);
//                img1.copyTo(show(cv::Rect(0, 0, img1.cols, img1.rows)));
//                img2.copyTo(show(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
//                if (min_id != -1 && min_dist < 100)
//                {
//                    if (frame2->features_depth[min_id] > 0) {
//                        cv::line(show, cv::Point((int)frame1->get_features_uv()[i].x(), (int)frame1->get_features_uv()[i].y()),
//                                 cv::Point((int)frame2->get_features_uv()[min_id].x() + img1.cols, (int)frame2->get_features_uv()[min_id].y()),
//                                 cv::Scalar(0, 0, 255), 2);
//                    }
//                }
//                cv::imshow("show", show);
//                cv::waitKey(0);
            }
        }

        // draw matches
//        cv::Mat img1 = frame1->image.clone();
//        cv::Mat img2 = frame2->image.clone();
//        if (img1.channels() == 1) {
//            cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
//        }
//        if (img2.channels() == 1) {
//            cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);
//        }
//
//        for (int i = 0; i < frame1->get_features_uv().size(); i++) {
//            if (frame1->features_depth[i] > 0) {
//                cv::circle(img1, cv::Point((int)frame1->get_features_uv()[i].x(), (int)frame1->get_features_uv()[i].y()), 2, cv::Scalar(0, 255, 0), 2);
//            }
//        }
//        for (int i = 0; i < frame2->get_features_uv().size(); i++) {
//            if (frame2->features_depth[i] > 0) {
//                cv::circle(img2, cv::Point((int)frame2->get_features_uv()[i].x(), (int)frame2->get_features_uv()[i].y()), 2, cv::Scalar(0, 255, 0), 2);
//            }
//        }
////        cv::imshow("img1", img1);
////        cv::imshow("img2", img2);
//
//        cv::Mat show = cv::Mat(std::max(img1.rows, img2.rows), img1.cols + img2.cols, CV_8UC3);
//        img1.copyTo(show(cv::Rect(0, 0, img1.cols, img1.rows)));
//        img2.copyTo(show(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
//
//        for (auto match : matches) {
//            cv::line(show, cv::Point((int)frame1->get_features_uv()[match.first].x(), (int)frame1->get_features_uv()[match.first].y()),
//                     cv::Point((int)frame2->get_features_uv()[match.second].x() + img1.cols, (int)frame2->get_features_uv()[match.second].y()),
//                     cv::Scalar(0, 0, 255), 2);
//        }
//        cv::imshow("show", show);
//        cv::waitKey(0);

        return matches;
    }
    void Matcher::robust_ratio_test(std::vector<std::vector<cv::DMatch> >& matches) {

        // Loop through all matches
        for(auto matchIterator=matches.begin(); matchIterator!= matches.end(); ++matchIterator)
        {
            // If 2 NN has been identified, else remove this feature
            if (matchIterator->size() > 1) {
                // check distance ratio, remove it if the ratio is larger
                if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > 0.8 ) {
                    matchIterator->clear();
                }
            } else {
                // does not have 2 neighbours, so remove it
                matchIterator->clear();
            }
        }
    
    }

    void Matcher::robust_symmetry_test(std::vector<std::vector<cv::DMatch> >& matches1,
                                           std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& good_matches) {

    std::cout <<"in TrackDescriptor::robust_symmetry_test"<< std::endl;
    // for all matches image 1 -> image 2
    for (auto matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1) {

        // ignore deleted matches
        if (matchIterator1->empty() || matchIterator1->size() < 2)
            continue;

        // for all matches image 2 -> image 1
        for (auto matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2) {
            // ignore deleted matches
            if (matchIterator2->empty() || matchIterator2->size() < 2)
                continue;

            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx && (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
                // add symmetrical match
                good_matches.emplace_back(cv::DMatch((*matchIterator1)[0].queryIdx,(*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance));
                // next match in image 1 -> image 2
                break;
            }
        }
    }

}

std::vector<std::pair<int, int>> Matcher::match_stereo(
    cv::Mat img1, cv::Mat img2,
    std::vector<Eigen::Vector2d> &features_uv_left,
    std::vector<Eigen::Vector2d> &features_uv_right,
    cv::Mat &descriptors_left, cv::Mat &descriptors_right) {
std::vector<std::pair<int, int>> matches;
std::vector<cv::DMatch> bfmatches;
// 1. brute force matching
cv::BFMatcher matcher(cv::NORM_L2, true);
matcher.match(descriptors_left, descriptors_right, bfmatches);
// 5. fundamental matrix estimation
std::vector<cv::Point2f> points1, points2;
for (auto& match : bfmatches)
{
    //points1.push_back(key_points[match.queryIdx].pt);
    points1.push_back(cv::Point2f(features_uv_left[match.queryIdx].x(), features_uv_left[match.queryIdx].y()));
    //points2.push_back(key_points2[match.trainIdx].pt);
    points2.push_back(cv::Point2f(features_uv_right[match.trainIdx].x(), features_uv_right[match.trainIdx].y()));
}
std::vector<uchar> status;
cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99, status);

std::vector<cv::DMatch> good_matches;
std::vector<cv::DMatch> bad_matches;
for (int i = 0; i < status.size(); i++)
{
    if (status[i] == 1)
    {
        good_matches.push_back(bfmatches[i]);
        std::pair<int,int> onepair;
        onepair.first = bfmatches[i].queryIdx;
        onepair.second = bfmatches[i].trainIdx;
        matches.push_back(onepair);
    }
    else
    {
        bad_matches.push_back(bfmatches[i]);
    }
}

// std::ofstream ofs1("matches.txt");
// ofs1 << std::setprecision(15);
// for (const auto m:good_matches) {
//     ofs1<<m.queryIdx << " "<<m.trainIdx<<std::endl;

// }


//    std::vector<cv::KeyPoint> features_left, features_right;
//    for (auto &uv : features_uv_left) {
//        cv::KeyPoint kp((float)uv.x(), (float)uv.y(), 1.0f, -1, 0, 0, -1);
//        features_left.push_back(kp);
//    }
//    for (auto &uv : features_uv_right) {
//        cv::KeyPoint kp((float)uv.x(), (float)uv.y(), 1.0f, -1, 0, 0, -1);
//        features_right.push_back(kp);
//    }
//    cv::Mat show;
//    cv::drawMatches(img1, features_left, img2, features_right, good_matches, show);
//    cv::imshow("Matches", show);
//    cv::waitKey(1000);

return matches;
}
//     std::vector<std::pair<int, int>> Matcher::match_stereo(
//             cv::Mat img1, cv::Mat img2,
//             std::vector<Eigen::Vector2d> &features_uv_left,
//             std::vector<Eigen::Vector2d> &features_uv_right,
//             cv::Mat &descriptors_left, cv::Mat &descriptors_right) {
//         std::vector<std::pair<int, int>> matches;
//         // // 1. brute force matching
//         // cv::BFMatcher matcher(cv::NORM_L2);
//         // std::vector<std::vector<cv::DMatch>> knn_matches;
//         // std::vector<cv::DMatch> good_matches;
//         // matcher.knnMatch(descriptors_left, descriptors_right, knn_matches, 2);
//         // // 2. filter matches
//         // for (int i = 0; i < knn_matches.size(); i++) {
//         //     if (knn_matches[i][0].distance < 0.8 * knn_matches[i][1].distance) {
//         //         matches.emplace_back(i, knn_matches[i][0].trainIdx);
//         //         good_matches.push_back(knn_matches[i][0]);
//         //     }
//         // }



//         // Our 1to2 and 2to1 match vectors
//         std::vector<std::vector<cv::DMatch> > matches0to1, matches1to0;
//         cv::BFMatcher matcher(cv::NORM_L2);
//         matcher.knnMatch(descriptors_left, descriptors_right, matches0to1, 2);
//         matcher.knnMatch(descriptors_right, descriptors_left, matches1to0, 2);
    
//         robust_ratio_test(matches0to1);
//         robust_ratio_test(matches1to0);


//     // Finally do a symmetry test
//     std::vector<cv::DMatch> matches_good;
//     robust_symmetry_test(matches0to1, matches1to0, matches_good);


//         // Convert points into points for RANSAC
//         std::vector<cv::Point2f> pts0_rsc, pts1_rsc;
//         for(size_t i=0; i<matches_good.size(); i++) {
//             // Get our ids
//             int index_pt0 = matches_good.at(i).queryIdx;
//             int index_pt1 = matches_good.at(i).trainIdx;
//             // Push back just the 2d point
//             pts0_rsc.push_back(cv::Point2f(features_uv_left[index_pt0].x(), features_uv_left[index_pt0].y()));
//             pts1_rsc.push_back(cv::Point2f(features_uv_right[index_pt1].x(), features_uv_right[index_pt1].y()));
//         }
    
//         // If we don't have enough points for ransac just return empty
//         if(pts0_rsc.size() < 10)
//             return matches;

        
//     // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
//     std::vector<uchar> mask_rsc;
//     // double max_focallength_img0 = std::max(camera_k_OPENCV.at(id0)(0,0),camera_k_OPENCV.at(id0)(1,1));
//     // double max_focallength_img1 = std::max(camera_k_OPENCV.at(id1)(0,0),camera_k_OPENCV.at(id1)(1,1));
//     // double max_focallength = std::max(max_focallength_img0,max_focallength_img1);
//     cv::findFundamentalMat(pts0_rsc, pts1_rsc, cv::FM_RANSAC, 2, 0.999, mask_rsc);
//         // Loop through all good matches, and only append ones that have passed RANSAC
//         for(size_t i=0; i<matches_good.size(); i++) {
//             // Skip if bad ransac id
//             if (mask_rsc[i] != 1)
//                 continue;
//             // Else, lets append this match to the return array!
//             matches.emplace_back(matches_good.at(i).queryIdx, matches_good.at(i).trainIdx);
//         }
    

// //    std::vector<cv::KeyPoint> features_left, features_right;
// //    for (auto &uv : features_uv_left) {
// //        cv::KeyPoint kp((float)uv.x(), (float)uv.y(), 1.0f, -1, 0, 0, -1);
// //        features_left.push_back(kp);
// //    }
// //    for (auto &uv : features_uv_right) {
// //        cv::KeyPoint kp((float)uv.x(), (float)uv.y(), 1.0f, -1, 0, 0, -1);
// //        features_right.push_back(kp);
// //    }
// //    cv::Mat show;
// //    cv::drawMatches(img1, features_left, img2, features_right, good_matches, show);
// //    cv::imshow("Matches", show);
// //    cv::waitKey(1000);

//         return matches;
//     }

    std::vector<std::pair<int, int>> Matcher::match_knn(Frame& frame1, Frame& frame2) {
        std::vector<std::pair<int, int>> matches;
        // 1. brute force matching
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        std::vector<cv::DMatch> good_matches;
        matcher.knnMatch(frame1.descriptors, frame2.descriptors, knn_matches, 2);
        // 2. filter matches
        for (int i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < 0.8 * knn_matches[i][1].distance) {
                matches.emplace_back(i, knn_matches[i][0].trainIdx);
                good_matches.push_back(knn_matches[i][0]);
            }
        }
         // Our 1to2 and 2to1 match vectors
    //      std::vector<std::vector<cv::DMatch> > matches0to1, matches1to0;
    //      cv::BFMatcher matcher(cv::NORM_L2);
    //      std::cout<<"frame1.descriptors.size():"<<frame1.descriptors.size()<<std::endl;
    //      std::cout<<"frame2.descriptors.size():"<<frame2.descriptors.size()<<std::endl;
    //      matcher.knnMatch(frame1.descriptors,  frame2.descriptors, matches0to1, 2);
    //      matcher.knnMatch( frame2.descriptors, frame1.descriptors, matches1to0, 2);
     
    //      robust_ratio_test(matches0to1);
    //      robust_ratio_test(matches1to0);
 
 
    //  // Finally do a symmetry test
    //  std::vector<cv::DMatch> matches_good;
    //  robust_symmetry_test(matches0to1, matches1to0, matches_good);
    //  std::cout<<"matches_good.descriptors.size():"<<matches_good.size()<<std::endl;
 
    //      // Convert points into points for RANSAC
    //      std::vector<cv::Point2f> pts0_rsc, pts1_rsc;
    //      for(size_t i=0; i<matches_good.size(); i++) {
    //          // Get our ids
    //          int index_pt0 = matches_good.at(i).queryIdx;
    //          int index_pt1 = matches_good.at(i).trainIdx;
    //          // Push back just the 2d point
    //          pts0_rsc.push_back(cv::Point2f(frame1.features_uv[index_pt0].x(), frame1.features_uv[index_pt0].y()));
    //          pts1_rsc.push_back(cv::Point2f(frame2.features_uv[index_pt1].x(), frame2.features_uv[index_pt1].y()));
    //      }
     
    //      // If we don't have enough points for ransac just return empty
    //      if(pts0_rsc.size() < 10)
    //          return matches;
 
         
    //  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    //  std::vector<uchar> mask_rsc;
    //  // double max_focallength_img0 = std::max(camera_k_OPENCV.at(id0)(0,0),camera_k_OPENCV.at(id0)(1,1));
    //  // double max_focallength_img1 = std::max(camera_k_OPENCV.at(id1)(0,0),camera_k_OPENCV.at(id1)(1,1));
    //  // double max_focallength = std::max(max_focallength_img0,max_focallength_img1);
    //  cv::findFundamentalMat(pts0_rsc, pts1_rsc, cv::FM_RANSAC, 5, 0.999, mask_rsc);
    //      // Loop through all good matches, and only append ones that have passed RANSAC
    //      for(size_t i=0; i<matches_good.size(); i++) {
    //          // Skip if bad ransac id
    //          if (mask_rsc[i] != 1)
    //              continue;
    //          // Else, lets append this match to the return array!
    //          matches.emplace_back(matches_good.at(i).queryIdx, matches_good.at(i).trainIdx);
    //      }
     
        return matches;
    }

    std::vector<std::pair<int, int>> Matcher::match_descriptor(cv::Mat descriptors1, cv::Mat descriptors2) {
        std::vector<std::pair<int, int>> matches;
        // 1. brute force matching
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        std::vector<cv::DMatch> good_matches;
        matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);
        // 2. filter matches
        for (int i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < 0.8 * knn_matches[i][1].distance) {
                matches.emplace_back(i, knn_matches[i][0].trainIdx);
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        return matches;
    }

    std::vector<std::pair<int, int>>
    Matcher::match_projective(Frame &frame, std::vector<std::shared_ptr<MapPoint>> map_points) {
        std::vector<std::pair<int, int>> matches;
        for (int i = 0;i < map_points.size();i ++) {
            std::shared_ptr<MapPoint> mp = map_points[i];
            if (mp != nullptr) {
                Eigen::Vector3d x3D = mp->x3D;
                Eigen::Vector3d x3D_2 = frame.get_R().transpose() * (x3D - frame.get_t());
                Eigen::Vector2d uv = frame.get_camera()->project(x3D_2);
                std::vector<int> around_ids = frame.get_around_features(uv, 50);
                cv::Mat desc_1 = mp->descriptor;
                double min_dist = 1000;
                int min_id = -1;
                for (int around_id : around_ids) {
                    cv::Mat desc_2 = frame.descriptors.row(around_id);
                    double dist = cv::norm(desc_1, desc_2);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_id = around_id;
                    }
                }
                if (min_id != -1 && min_dist < 100) {
                    matches.emplace_back(min_id, i);
                }
            }
        }
        return matches;
    }
}

