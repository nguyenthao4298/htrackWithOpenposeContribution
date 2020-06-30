#include "Sensor.h"
#include "tracker/Types.h"

#include "util/gl_wrapper.h"
#include "util/mylogger.h"
#include "util/Sleeper.h"
#include "util/tictoc.h"

#ifndef HAS_LIBFREENECT
SensorLibFreenect::SensorLibFreenect(Camera *camera) : Sensor(camera) { mFatal() << "Intel Libfreenect not available in your OS"; }
int SensorLibFreenect::initialize() { return 0; }
SensorLibFreenect::~SensorLibFreenect() {}
bool SensorLibFreenect::spin_wait_for_data(Scalar timeout_seconds) { return false; }
bool SensorLibFreenect::fetch_streams(DataFrame &frame) { return false; }
void SensorLibFreenect::start() {}
void SensorLibFreenect::stop() {}
#else
#include "Sensor.h"
#include <stdio.h>
#include <vector>
#include <exception>
#include <time.h>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <iostream>
#include <limits>
#include <QElapsedTimer>
#include <QApplication>
#include <QMessageBox>
#include "tracker/Data/DataFrame.h"
#include "tracker/Data/Camera.h"

#include <openpose/flags.hpp>
#include <openpose/headers.hpp>
using namespace std;

int D_width  = 960;
int D_height = 540;
//! [context]
libfreenect2::Freenect2 freenect2;
libfreenect2::Freenect2Device *dev = 0;
libfreenect2::PacketPipeline *pipeline = 0;
libfreenect2::SyncMultiFrameListener *listener;
libfreenect2::Registration *registration;
//! [OpenPose here]
op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
void configureWrapper(op::Wrapper &opWrapper)
{
  try
  {
    // Configuring OpenPose

    // logging_level
    op::checkBool(
        0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
        __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    op::Profiler::setDefaultX(FLAGS_profile_speed);

    // Applying user defined configuration - GFlags to program variables
    // outputSize
    const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
    // faceNetInputSize
    const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
    // handNetInputSize
    const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
    // poseMode
    const auto poseMode = op::flagsToPoseMode(FLAGS_body);
    // poseModel
    const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
    // JSON saving
    if (!FLAGS_write_keypoint.empty())
      op::opLog(
          "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
          " instead.",
          op::Priority::Max);
    // keypointScaleMode
    const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
    // heatmaps to add
    const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                  FLAGS_heatmaps_add_PAFs);
    const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
    // >1 camera view?
    const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
    // Face and hand detectors
    const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
    const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
    // Enabling Google Logging
    const bool enableGoogleLogging = true;

    // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
    const op::WrapperStructPose wrapperStructPose{
        poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
        FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
        poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
        FLAGS_part_to_show, op::String(FLAGS_model_folder), heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
        (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
        op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
        (float)FLAGS_upsampling_ratio, enableGoogleLogging};
    opWrapper.configure(wrapperStructPose);
    //Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{
        FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
        op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
        (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
    opWrapper.configure(wrapperStructHand);
    // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
    const op::WrapperStructExtra wrapperStructExtra{
        FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
    opWrapper.configure(wrapperStructExtra);
    // Output (comment or use default argument to disable any output)
    const op::WrapperStructOutput wrapperStructOutput{
        FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
        op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
        FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
        op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
        op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
        op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
        op::String(FLAGS_udp_port)};
    opWrapper.configure(wrapperStructOutput);
    // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
    // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
    if (FLAGS_disable_multi_thread)
      opWrapper.disableMultiThreading();
  }
  catch (const std::exception &e)
  {
    op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
  }
}
// ! [OpenPose here]
// ! [context]
SensorLibFreenect::SensorLibFreenect(Camera *camera) : Sensor(camera)
{
  if (camera->mode() != Kinect)
    LOG(FATAL) << "!!!FATAL: LibFreenect needs kinect camera mode";
}
int SensorLibFreenect::initialize()
{
  //FLAGS_hand = true;
  //FLAGS_hand_net_resolution = "256x256";
  FLAGS_net_resolution = "-1x160";
  configureWrapper(opWrapper);
  opWrapper.start();
  // //! [OpenPose here]
  std::cout << "this is SensorLibFreenect::initialize()" << std::endl;
  //! [context]
  if (freenect2.enumerateDevices() == 0)
  {
    std::cout << "No device connected!" << std::endl;
    return EXIT_FAILURE;
  }

  std::string serial = freenect2.getDefaultDeviceSerialNumber();

  // std::cout << "SERIAL: " << serial << std::endl;

  if (pipeline)
  {
    //! [open]Wrist
  }
  else
  {
    dev = freenect2.openDevice(serial);
  }

  if (dev == 0)
  {
    std::cout << "failure opening device!" << std::endl;
    return EXIT_FAILURE;
  }
  listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
  dev->setColorFrameListener(listener);
  dev->setIrAndDepthFrameListener(listener);

  dev->start();
  registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
  /// [start]
  dev->start();
  printf("Device Started\n");
  this->initialized = true;
  return true;
}
SensorLibFreenect::~SensorLibFreenect()
{
  std::cout << "~SensorLibFreenect()" << std::endl;
  opWrapper.stop();
  if (!initialized)
  std::cout << "dddd" <<std::endl;
    return;
  //TODO: stop sensor
}

bool SensorLibFreenect::spin_wait_for_data(Scalar timeout_seconds)
{
  DataFrame frame(-1);
  QElapsedTimer chrono;
  chrono.start();
  while (fetch_streams(frame) == false)
  {
    LOG(INFO) << "Waiting for data.. " << chrono.elapsed();
    Sleeper::msleep(500);
    QApplication::processEvents(QEventLoop::AllEvents);
    if (chrono.elapsed() > 1000 * timeout_seconds)
      return false;
  }
  return true;
}

bool SensorLibFreenect::fetch_streams(DataFrame &frame)
{
  // clock_t t;
  // t = clock();
  // printf("Timer starts\n");
  if (initialized == false)
    this->initialize();

  if (frame.depth.empty())
    frame.depth = cv::Mat(cv::Size(D_width, D_height), CV_16UC1, cv::Scalar(0));
  if (frame.color.empty())
    frame.color = cv::Mat(cv::Size(D_width, D_height), CV_8UC3, cv::Scalar(0, 0, 0));
  libfreenect2::FrameMap frames;
  listener->waitForNewFrame(frames);

  libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
  //libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
  libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4); // check here (https://github.com/OpenKinect/libfreenect2/issues/337) and here (https://github.com/OpenKinect/libfreenect2/issues/464) why depth2rgb image should be bigger

  //-----fail
  registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);
  // cv::Mat depthmat =cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
  cv::Mat rgbmat = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
  cv::Mat bigdepth = cv::Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data);
  cv::Mat rgb1, depth1;
  cv::resize(rgbmat, rgbmat, cv::Size(960, 540));
  cv::resize(bigdepth, bigdepth, cv::Size(960, 540));
  rgbmat.convertTo(rgb1, CV_8UC3);
  bigdepth.convertTo(depth1, CV_16UC1);
  
  //bigDepthMat.convertTo(bigDepthMat,CV_8UC1,255.0);
  cv::Mat depth_buffer = depth1;
  cv::Mat color_buffer = rgb1;
  //-----fail
  // cv::Mat depth_buffer = cv::Mat(cv::Size(D_width, D_height), CV_16UC1, cv::Scalar(0));
  // cv::Mat color_buffer = cv::Mat(cv::Size(D_width, D_height), CV_8UC3, cv::Scalar(255, 255, 255));
  // for (int dy = 0, dy_sub = 0; dy < depth->height; dy += 1, dy_sub++)
  // {
  //   for (int dx = 0, dx_sub = 0; dx < depth->width; dx += 1, dx_sub++)

  //   {
  //     float X, Y, Z, colors;
  //     registration->getPointXYZRGB(&undistorted, &registered, dy, dx, X, Y, Z, colors);
  //     if (std::isnan(Z))
  //       continue;
  //     float depth_value_in_mm = Z * 1000;
  //     const int cx = (int)std::round(X), cy = (int)std::round(Y);
  //     if (cx < 0 || cy < 0 || cx >= (rgb->height) || cy >= (rgb->width))
  //     {
  //       color_buffer.at<cv::Vec3b>(dy_sub, dx_sub) = cv::Vec3b(255, 255, 255);
  //     }
  //     else
  //     {
  //       const uint8_t *p = reinterpret_cast<uint8_t *>(&colors);
  //       unsigned char b = p[0];
  //       unsigned char g = p[1];
  //       unsigned char r = p[2];
  //       color_buffer.at<cv::Vec3b>(dy_sub, dx_sub) = cv::Vec3b(r, g, b);
  //       depth_buffer.at<unsigned short>(dy_sub, dx_sub) = (unsigned short)depth_value_in_mm;
  //     }
  //   }
  // }
  // cv::Mat rgbd(registered.height, registered.width, CV_8UC4, registered.data);
  cv::cvtColor(rgbmat, rgbmat, CV_BGRA2BGR);
  const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(rgbmat);
  auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
  if (datumProcessed != nullptr)
  {
    //const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
    //cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);

    const auto numberPeopleDetected = datumProcessed->at(0)->poseKeypoints.getSize(0);
    //std::cout << numberPeopleDetected << std::endl;
    const auto numberBodyParts = datumProcessed->at(0)->poseKeypoints.getSize(1);
    if (numberPeopleDetected != 0)
    {

      const auto baseIndexwrist = datumProcessed->at(0)->poseKeypoints.getSize(2) * (0 * numberBodyParts + 4);
      //const auto baseIndexelbow = datumProcessed->at(0)->poseKeypoints.getSize(2) * (0 * numberBodyParts + 3);
      const auto xwrist = datumProcessed->at(0)->poseKeypoints[baseIndexwrist];
      const auto ywrist = datumProcessed->at(0)->poseKeypoints[baseIndexwrist + 1];
      //const auto xelbow = datumProcessed->at(0)->poseKeypoints[baseIndexelbow];
      //const auto yelbow = datumProcessed->at(0)->poseKeypoints[baseIndexelbow + 1];
      // Camera *const newCam = (Camera *const)camera;
      // frame.wrist = frame.point_at_pixel( ywrist,xwrist, newCam);
      //std::cout << " vi tri x " << xwrist << " va " << ywrist <<std::endl;
      // if (xwrist == 0 && ywrist == 0)
      // {
      //   std::cout << "khong nhan duoc co tay dau"<<std::endl;
      // }
      cv::Mat rgbd2;
      cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo(rgbd2);
      cv::cvtColor(rgbd2, rgbd2, CV_BGRA2BGR);
      cv::resize(rgbd2, rgbd2, cv::Size(960, 540));
      cv::Point startPt(xwrist, ywrist);
      frame.wrist_location = Vector2(xwrist, ywrist);
      // cv::Mat fake1 = rgbd;
      //cv::circle(rgbd2, startPt, 5, cv::Scalar(123, 255, 222), CV_FILLED, 4);
      //cv::inRange(rgbd2, cv::Scalar(123, 255, 222), cv::Scalar(123, 255, 222), fake1);
      cv::imshow("", rgbd2);
      // std::cout << " dam bao 1" << frame.wrist << " oke " << std::endl;
      // cv::Point startPt(xwrist, ywrist);
      // //cv::line(rgbmat, cv::Point(xelbow, yelbow), cv::Point(xwrist, ywrist), cv::Scalar(255, 255, 0), 4);
      // float vX = xwrist - xelbow;
      // float vY = ywrist - yelbow;
      // if ((vX != 0) || (vY != 0))
      // {
      //   float mag = sqrt(vX * vX + vY * vY);
      //   vX = vX / mag;
      //   vY = vY / mag;
      //   float temp = vX;
      //   vX = 0 - vY;
      //   vY = temp;
      //   float cX = xwrist + vX * 30;
      //   float cY = ywrist + vY * 30;
      //   float dX = xwrist - vX * 30;
      //   float dY = ywrist - vY * 30;
      //   //cv::line(rgbmat, cv::Point(cX, cY), cv::Point(dX, dY), cv::Scalar(255, 0, 0), 130);
      //   // cv::line(rgbd, cv::Point(cX, cY), cv::Point(dX, dY), cv::Scalar(255, 255, 255), 20);
      //   cv::circle(rgbd, startPt, 10, cv::Scalar(0, 255, 0), 4);
      //   cv::imshow("hello", rgbd);
      // }
      //cv::circle(rgbmat, startPt, 10, cv::Scalar(0, 255, 0), 4);
      //cv::imshow("hello", masked);
    }
    else
    {
      std::cout << "can't find hand in Openpose !" << std::endl;
    }
  }
  else
  {
    op::opLog("Image could not be processed.", op::Priority::High);
  }

  frame.color = color_buffer;
  frame.depth = depth_buffer;
  listener->release(frames);
  // printf("Timer ends \n");
  // t = clock() - t;
  // double time_taken = ((double)t) / CLOCKS_PER_SEC; // calculate the elapsed time
  // std::cout << "The program took " << time_taken << " seconds to execute" << std::endl;
  return true;
}
void SensorLibFreenect::start()
{
  if (!initialized)
    this->initialize();
}

void SensorLibFreenect::stop()
{
  dev->stop();
  dev->close();
}

#endif
