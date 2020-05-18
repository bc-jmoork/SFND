#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void gradient_sobel()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");

    // convert image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // create filter kernel
    float sobel_x[9] = {-1, 0, +1,
                        -2, 0, +2, 
                        -1, 0, +1};
    cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_x);
    float sobel_y[9] = {-1, -2, -1,
                        0., 0.,  0., 
                        +1, +2, +1};
    cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);

    // apply filter
    cv::Mat blurred, result_x, result_y;
    int kernel_size = 5;
    float stddev = 2.0;
    cv::GaussianBlur(imgGray, blurred, cv::Size(kernel_size, kernel_size), stddev);
    cv::filter2D(blurred, result_x, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(blurred, result_y, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::Mat grad_mag = imgGray.clone();
    for (auto r = 0; r < grad_mag.rows; r++) {
        for (auto c = 0; c < grad_mag.cols; c++) {
            grad_mag.at<unsigned char>(r, c) = sqrt(pow(result_x.at<unsigned char>(r, c), 2) + 
                                                    pow(result_y.at<unsigned char>(r, c), 2));
        }
    }

    // show result
    string windowName = "Sobel operator (x-direction)";
    cv::namedWindow( windowName, 1 ); // create window 
    cv::imshow(windowName, grad_mag);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    gradient_sobel();
}
