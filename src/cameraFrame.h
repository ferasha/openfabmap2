/*
 * cameraFrame.h
 *
 *  Created on: Aug 4, 2016
 *      Author: rasha
 */

#ifndef CAMERAFRAME_H_
#define CAMERAFRAME_H_

#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

class cameraFrame {
public:
	cameraFrame();
	virtual ~cameraFrame();

	cameraFrame(cv_bridge::CvImagePtr& cv_ptr);
	cameraFrame(cv_bridge::CvImagePtr& cv_img_ptr, cv_bridge::CvImagePtr& cv_depth_ptr, const sensor_msgs::CameraInfoConstPtr& cam_info_ptr);
	cameraFrame(cv::Mat& depth_img);

	cv_bridge::CvImagePtr image_ptr;
	cv_bridge::CvImagePtr depth_ptr;
	float fx, fy, cx, cy;
	cv::Mat depth_img;
};

#endif /* CAMERAFRAME_H_ */
