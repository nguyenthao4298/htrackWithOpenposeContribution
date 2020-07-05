#pragma once
#include "tracker/ForwardDeclarations.h"
#include "tracker/Types.h"
#include "util/opencv_wrapper.h"
#include "tracker/Detection/TrivialDetector.h"
class HandFinder{
private:
    Camera*const camera=NULL;
    TrivialDetector*const trivial_detector=NULL;
public:
    HandFinder(Camera*, TrivialDetector*);

/// @{ Settings
public:
    struct Settings{
        bool show_hand = false;
        bool show_wband = false;
        float depth_range = 150;
        float wband_size = 30;
        cv::Scalar hsv_min = cv::Scalar( 94, 111,  37); ///< potentially read from file
        cv::Scalar hsv_max = cv::Scalar(120, 255, 255); ///< potentially read from file
    } _settings;
    Settings*const settings=&_settings;
/// @}

private:
    bool _has_useful_data = false;
    bool _wrist_found;
    Vector3 _wrist_center;
    Vector3 _wrist_dir;
public:
    cv::Mat sensor_silhouette; ///< created by binary_classifier
    cv::Mat sensor_wristband; ///< created by binary_classifier --no more using with the contribution of Openpose

public:
    bool has_useful_data(){ return _has_useful_data; }
    bool wrist_found(){ return _wrist_found; }
    Vector3 wrist_center(){ return _wrist_center; }
    Vector3 wrist_direction(){ return _wrist_dir; }
    void wrist_direction_flip(){ _wrist_dir=-_wrist_dir; }
public:
    void binary_classification(DataFrame &frame);
};
