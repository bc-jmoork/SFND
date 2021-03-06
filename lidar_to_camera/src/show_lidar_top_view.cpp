#include <iostream>
#include <numeric>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "structIO.hpp"

using namespace std;

// A pred function to adjust according to your score
bool comparator_x(const LidarPoint& s1, const LidarPoint& s2) {
   return s1.x < s2.x;
}

// A pred function to adjust according to your score
bool comparator_y(const LidarPoint& s1, const LidarPoint& s2) {
   return s1.y < s2.y;
}

void showLidarTopview()
{
    std::vector<LidarPoint> lidarPoints;
    readLidarPts("../dat/C51_LidarPts_0000.dat", lidarPoints);
    auto max_point = std::max_element(lidarPoints.begin(), lidarPoints.end(), comparator_x);
    std::cout << max_point->x << "," << max_point->y << endl;
    auto max_point_y = std::max_element(lidarPoints.begin(), lidarPoints.end(), comparator_y);
    std::cout << max_point_y->x << "," << max_point_y->y << endl;

    cv::Size worldSize(10.0, 20.0); // width and height of sensor field in m
    cv::Size imageSize(1000, 2000); // corresponding top view image in pixel

    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));

    // plot Lidar points into image
    cout << imageSize.height << "," << imageSize.width << endl;
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {
        float xw = (*it).x; // world position in m with x facing forward from sensor
        float yw = (*it).y; // world position in m with y facing left from sensor
        float zw = (*it).z; // world position in m from the ground

        int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-yw * imageSize.height / worldSize.height) + imageSize.width / 2;

        // TODO: 
        // 1. Change the color of the Lidar points such that 
        // X=0.0m corresponds to red while X=20.0m is shownas green.
        int maxvval = worldSize.height;
        int r = int(((maxvval - fmin(xw, maxvval)) / maxvval) * 255);
        int g = int ((fmin(xw, maxvval) / maxvval) * 255);

        // 2. Remove all Lidar points on the road surface while preserving 
        // measurements on the obstacles in the scene.
        float min_z = 1.40;
        zw = zw + min_z;
        if (zw >= 0.0)
            cv::circle(topviewImg, cv::Point(x, y), 5, cv::Scalar(0, g, r), -1);
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
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0); // wait for key to be pressed
}

int main()
{
    showLidarTopview();
}