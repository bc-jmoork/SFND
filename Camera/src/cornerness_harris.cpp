#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    cv::Mat thresh_dst;
    thresh_dst = cv::Mat::zeros(dst_norm_scaled.size(), CV_8UC1);
    std::vector<cv::KeyPoint> key_points;

    for(auto r=0; r < dst_norm_scaled.rows; r++) {
        for(auto c=0; c< dst_norm_scaled.cols; c++) {
            int response = (int) dst_norm_scaled.at<unsigned char>(r, c);
            if (response > minResponse) {
                cv::KeyPoint my_keypoint;
                my_keypoint.pt = cv::Point(c, r);
                my_keypoint.response = response;
                my_keypoint.size = 2*apertureSize;

                // go overall the elements in the key-point and remove the
                // keypoint if they overlap and the new one has larger response
                bool insert_keypoint = true;
                for(auto itr=key_points.begin(); itr < key_points.end(); itr++) {
                    // found a keypoint with overlap
                    if ((cv::KeyPoint::overlap(my_keypoint, *itr) > 0) and (my_keypoint.response > (*itr).response)) {
                        thresh_dst.at<unsigned char>(itr->pt.y, itr->pt.x) = 0.;
                        insert_keypoint = false;
                        thresh_dst.at<unsigned char>(my_keypoint.pt.y, my_keypoint.pt.x) = 255;
                        *itr = my_keypoint;
                        break;
                    }
                }

                if (insert_keypoint) {
                    key_points.push_back(my_keypoint);
                    thresh_dst.at<unsigned char>(my_keypoint.pt.y, my_keypoint.pt.x) = 255;
                }
            }
        }
    }
    cout << "Total keypoints: " << key_points.size() << endl;

    windowName = "Harris Corner NMS";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, thresh_dst);
    cv::waitKey(0);

}

int main()
{
    cornernessHarris();
} 