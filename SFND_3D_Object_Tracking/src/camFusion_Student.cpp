
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, 
std::vector<LidarPoint> &lidarPoints, 
float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}

void draw_boxes(cv::Mat& visImg, cv::Rect& roi, std::string label) 
{
    // Draw rectangle displaying the bounding box
    int top, left, width, height;
    top = roi.y;
    left = roi.x;
    width = roi.width;
    height = roi.height;
    cv::rectangle(visImg, cv::Point(left, top), 
    cv::Point(left+width, top+height), cv::Scalar(0, 255, 0), 2);
    
    // Display label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(visImg, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);

    return;
}

// returns the bounding box that matches the bounding box id
std::vector<BoundingBox>::iterator get_bbox_from_boxid(
    std::vector<BoundingBox>& bbox, 
    int boxid)
{
    for(auto it=bbox.begin(); it != bbox.end(); it++)  {
        if (it->boxID == boxid) {
            return it;
        }
    }
}

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1].second > v[i2].second;});

  return idx;
}

void matchBoundingBoxes(
    std::vector<cv::DMatch> &matches, 
    std::map<int, int> &bbBestMatches, 
    DataFrame &prevFrame, 
    DataFrame &currFrame)
{
    // for each of the keypoint matched, we need to
    // assign to a unique bounding box in 
    // the current frame and the bounding box
    // in the previous frame
    auto M = currFrame.boundingBoxes.size();
    auto N = prevFrame.boundingBoxes.size();
    map<int, std::vector<int>> bbox_count;
    for(auto i=0; i < N; i++) {
        bbox_count[i] = std::vector<int>(M, 0);
    }

    for(auto kptmat = currFrame.kptMatches.begin();
        kptmat != currFrame.kptMatches.end();
        kptmat++) {
        // find the current frame bounding box for the matching keypoint
        auto cur_kpt = currFrame.keypoints[kptmat->trainIdx];
        auto cur_box_id = -1;
        for (auto cur_box=currFrame.boundingBoxes.begin();
             cur_box != currFrame.boundingBoxes.end();
             cur_box++) {
            if (cur_box->roi.contains(cur_kpt.pt)) {
                cur_box->keypoints.push_back(cur_kpt); // add the keypoint to bounding box roi
                cur_box_id = cur_box->boxID;
            }
        }

        // find the prev. frame bounding box for the matching keypoint
        auto prev_kpt = prevFrame.keypoints[kptmat->queryIdx];
        auto prev_box_id = -1;
        for (auto prev_box=prevFrame.boundingBoxes.begin();
             prev_box != prevFrame.boundingBoxes.end();
             prev_box++) {
            if (prev_box->roi.contains(prev_kpt.pt)) {
                prev_box_id = prev_box->boxID;
            }
        }

        // if proper bounding box exists for the keypoint
        // in the current and previous frame, save the results
        // and increment the counter that map between 
        // current and prev frame bounding box
        if (cur_box_id >=0 and prev_box_id >= 0 ) {
            bbox_count[prev_box_id][cur_box_id] += 1;
        }
    }

    // for each bounding box in the previous frame
    // find the bounding box in the current frame
    // that has maximum number of keypoints
    vector<pair<BoundingBox*, int>> max_per_box_id;
    for (auto bbox=prevFrame.boundingBoxes.begin();
            bbox != prevFrame.boundingBoxes.end();
            bbox++) {
            std::vector<int>& v = bbox_count[bbox->boxID];
            int max_val = *std::max_element(v.begin(), v.end());
            max_per_box_id.push_back(pair<BoundingBox*, int>(&(*bbox), max_val)); // saves boxID and maximum kpts matched
    }
    
    // start from the maximum kpts match and go to the minimum value..
    // find the bounding box that has maximum number of matches
    // and save the results in the bbBestMatches
    // that maps between the bounding box Id from the previous frame
    // to the bounding box ID that best matches in the current frame
    // based on number of keypoint matches that falls within the bbox
    vector<pair<int, int>> sorted_box_ids;
    for (auto idx : sort_indexes(max_per_box_id)) {
        BoundingBox* bbox = max_per_box_id[idx].first;
        std::vector<int>& v = bbox_count[bbox->boxID];
        int max_idx = std::max_element(
            v.begin(), v.end()) - v.begin();
        
        if (bbox_count[bbox->boxID][max_idx] > 0) {
            sorted_box_ids.push_back(pair<int, int>(bbox->boxID, bbox_count[bbox->boxID][max_idx]));
            bbBestMatches[bbox->boxID] = max_idx;
            for (auto prev_bbox=prevFrame.boundingBoxes.begin();
                    prev_bbox != prevFrame.boundingBoxes.end();
                    prev_bbox++) {
                bbox_count[prev_bbox->boxID][max_idx] = 0; // clear the best match for the rest
            }
        }
    }

    // visualize results
    bool bVis = true;
    if (bVis)
    {
        auto cur_img = currFrame.cameraImg;
        auto prev_img = prevFrame.cameraImg;
        for(auto data: sorted_box_ids) {
            auto box_id = data.first;
            auto kpts_count = data.second;
            auto cur_box_id = bbBestMatches[box_id];
            auto curr_bbox_it = get_bbox_from_boxid(currFrame.boundingBoxes, cur_box_id);
            cv::Mat visImage = cur_img.clone();
            string label = cv::format("%d", cur_box_id);
            draw_boxes(visImage, curr_bbox_it->roi, label);
            string windowName = "BBox of current frame";
            cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
            imshow(windowName, visImage);

            visImage = prev_img.clone();
            auto prev_box_id = box_id;
            auto prev_bbox_it = get_bbox_from_boxid(prevFrame.boundingBoxes, prev_box_id);
            label = cv::format("%d", prev_box_id);
            draw_boxes(visImage, prev_bbox_it->roi, label);
            windowName = "BBox of prev frame";
            cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }
    }
}
