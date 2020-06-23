#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, 
                      std::vector<cv::KeyPoint> &kPtsRef, 
                      cv::Mat &descSource, 
                      cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, 
                      std::string descriptorType, 
                      std::string matcherType, 
                      std::string selectorType)
{
    float knn_threshold = 0.8;
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { 
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches);
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        vector<vector<cv::DMatch>> knn_matches;
        // k nearest neighbors (k=2)
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        for (auto knn_pair : knn_matches) {
            if ((knn_pair[0].distance / knn_pair[1].distance) < knn_threshold) {
                matches.push_back(knn_pair[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints,
                   cv::Mat &img, cv::Mat &descriptors, 
                   string descriptorType)
{
    // BRIEF, FREAK, AKAZE
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create(0);  
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();        
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();        
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();        
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // perform feature description
    double t = (double)cv::getTickCount();
    keypoints.clear();

    // Detector parameters
    int blockSize = 5;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 75; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, 
                        blockSize, apertureSize, k, 
                        cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for(auto r=0; r < dst_norm_scaled.rows; r++) {
        for(auto c=0; c< dst_norm_scaled.cols; c++) {
            int response = (int) dst_norm_scaled.at<unsigned char>(r, c);
            if (response > minResponse) {
                cv::KeyPoint my_keypoint;
                my_keypoint.pt = cv::Point(c, r);
                my_keypoint.response = response;
                my_keypoint.size = 2 * apertureSize;
                my_keypoint.class_id = 0;

                // go overall the elements in the key-point and remove the
                // keypoint if they overlap and the new one has larger response
                bool insert_keypoint = true;
                for(auto itr=keypoints.begin(); itr < keypoints.end(); itr++) {
                    // found a keypoint with overlap
                    if ((cv::KeyPoint::overlap(my_keypoint, *itr) > 0) and \
                        (my_keypoint.response > (*itr).response)) {
                        insert_keypoint = false;
                        *itr = my_keypoint;
                        break;
                    }
                }

                if (insert_keypoint) {
                    keypoints.push_back(my_keypoint);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "number of keypoints:" << keypoints.size() << endl;
    cout << "Harris Keypoint Detector" << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Keypoint Detector";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, 
                        cv::Mat &img, std::string detectorType, 
                        bool bVis)
{
    // perform feature description
    cv::Mat descriptors;
    double t = (double)cv::getTickCount();

    // select appropriate descriptor
    cv::Ptr<cv::Feature2D> feature_detector;

    if (detectorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        feature_detector = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        int nfeatures=500;
        float scaleFactor=1.2f;
        int nlevels=8;
        feature_detector = cv::ORB::create(nfeatures, scaleFactor, nlevels);
    }
    else if (detectorType.compare("FAST") == 0)
    {
        int threshold = 50;
        feature_detector = cv::FastFeatureDetector::create(threshold);
    }
    else if (detectorType.compare("AKAZE") == 0) {
         cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
         akaze->setThreshold(0.02 / 4);
         feature_detector = akaze;
    }
    else if (detectorType.compare("SIFT") == 0) {
         feature_detector = cv::xfeatures2d::SIFT::create(0);
    }

    feature_detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, 
                          cv::Scalar::all(-1));
                          //cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType;
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

