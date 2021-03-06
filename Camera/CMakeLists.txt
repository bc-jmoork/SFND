cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall" "-pedantic")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

find_package(OpenCV 4.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executables for exercises
add_executable (gaussian_smoothing src/gaussian_smoothing.cpp)
target_link_libraries (gaussian_smoothing ${OpenCV_LIBRARIES})

add_executable (gradient_sobel src/gradient_sobel.cpp)
target_link_libraries (gradient_sobel ${OpenCV_LIBRARIES})

#add_executable (magnitude_sobel src/magnitude_sobel.cpp)
#target_link_libraries (magnitude_sobel ${OpenCV_LIBRARIES})

add_executable (cornerness_harris src/cornerness_harris.cpp)
target_link_libraries (cornerness_harris ${OpenCV_LIBRARIES})

add_executable (detect_keypoints src/detect_keypoints.cpp)
target_link_libraries (detect_keypoints ${OpenCV_LIBRARIES})

add_executable (describe_keypoints src/describe_keypoints.cpp)
target_link_libraries (describe_keypoints ${OpenCV_LIBRARIES})

# Executables for exercise
add_executable (descriptor_matching src/descriptor_matching.cpp src/structIO.cpp)
target_link_libraries (descriptor_matching ${OpenCV_LIBRARIES})

