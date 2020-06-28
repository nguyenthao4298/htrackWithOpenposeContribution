
//! [headers]
#include <iostream>
#include <stdio.h>
#include <iomanip>
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
//! [headers]
#include <chrono>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>
#include <string>
#include <errno.h>

#include <cstdio>

// Also include GLFW to allow for graphical display
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

double yaw, pitch, lastX, lastY; int ml;
static void on_mouse_button(GLFWwindow * win, int button, int action, int mods)
{
    if(button == GLFW_MOUSE_BUTTON_LEFT) ml = action == GLFW_PRESS;
}
static double clamp(double val, double lo, double hi) { return val < lo ? lo : val > hi ? hi : val; }
static void on_cursor_pos(GLFWwindow * win, double x, double y)
{
    if(ml)
    {
        yaw = clamp(yaw - (x - lastX), -120, 120);
        pitch = clamp(pitch + (y - lastY), -80, 80);
    }
    lastX = x;
    lastY = y;
}

int main() 
{
    int D_width = 512;
int D_height = 424;

//! [context]
libfreenect2::Freenect2 freenect2;
libfreenect2::Freenect2Device *dev = 0;
libfreenect2::PacketPipeline *pipeline = 0;
libfreenect2::SyncMultiFrameListener *listener;
libfreenect2::Registration *registration;
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
    //! [open]
    dev = freenect2.openDevice(serial, pipeline);
    //! [open]
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
    // Open a GLFW window to display our output
    glfwInit();
    GLFWwindow * win = glfwCreateWindow(1280, 960, "librealsense tutorial #3", nullptr, nullptr);
    glfwSetCursorPosCallback(win, on_cursor_pos);
    glfwSetMouseButtonCallback(win, on_mouse_button);
    glfwMakeContextCurrent(win);
    while(!glfwWindowShouldClose(win))
    {
        // Wait for new frame data
        glfwPollEvents();
          libfreenect2::FrameMap frames;
  listener->waitForNewFrame(frames);
  libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
  libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
  libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
  cv::Mat rgbmat = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
  cv::Mat depthmat = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
  cv::Mat rgb1, depth1;
  rgbmat.convertTo(rgb1, CV_8UC3);
  depthmat.convertTo(depth1, CV_16UC1);
  const uint16_t *depth_image = (const uint16_t *)rgb1.data;
  const uint8_t *color_image = (const uint8_t *)depth1.data;
  //! [registration setup]
  libfreenect2::Registration *registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4); // check here (https://github.com/OpenKinect/libfreenect2/issues/337) and here (https://github.com/OpenKinect/libfreenect2/issues/464) why depth2rgb image should be bigger
  //! [registration setup]
  cv::Mat depthmatUndistorted, irmat, rgbd, depth_fullscale;
  //   //! [registration]
  registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);

        // Set up a perspective transform in a space that we can rotate by clicking and dragging the mouse
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60, (float)1280/960, 0.01f, 20.0f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(0,0,0, 0,0,1, 0,-1,0);
        glTranslatef(0,0,+0.5f);
        glRotated(pitch, 1, 0, 0);
        glRotated(yaw, 0, 1, 0);
        glTranslatef(0,0,-0.5f);

        // We will render our depth data as a set of points in 3D space
        glPointSize(2);
        glEnable(GL_DEPTH_TEST);
        glBegin(GL_POINTS);

        for(int dy=0; dy<depth1.rows; ++dy)
        {
            for(int dx=0; dx< depth1.cols; ++dx)
            {
                // Retrieve the 16-bit depth value and map it into a depth in meters
uint16_t depth_value = depth_image[dy * (depth1.cols) + dx];
      float depth_in_meters = depth_value * 55;
      float pixel_depth_in_mm = depth_value * 1000; // is depth_value in milimerter???
      if (depth_value == 0)
        continue;
               float X, Y, Z, colors;
      registration->getPointXYZRGB(&undistorted, &registered, dy, dx, X, Y, Z, colors); //is Z depth_value ???
      const int cx = (int)std::round(X), cy = (int)std::round(Y);

                // Use the color from the nearest color pixel, or pure white if this point falls outside the color image
                if(cx < 0 || cy < 0 || cx >= rgb1.cols || cy >= rgb1.rows)
                {
                    glColor3ub(255, 255, 255);
                }
                else
                {
                    glColor3ubv(color_image + (cy * rgb1.cols + cx) * 3);
                }

                // Emit a vertex at the 3D location of this depth pixel
                glVertex3f(X, Y, Z);
            }
        }
  listener->release(frames);
        glEnd();

        glfwSwapBuffers(win);
    }

    return EXIT_SUCCESS;
}

