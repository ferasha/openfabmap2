/*
 * cameraFrame.cpp
 *
 *  Created on: Aug 4, 2016
 *      Author: rasha
 */

#include "cameraFrame.h"

cameraFrame::cameraFrame() {
	// TODO Auto-generated constructor stub

}

cameraFrame::cameraFrame(cv_bridge::CvImagePtr& cv_ptr)
{
		image_ptr = cv_ptr;
}
cameraFrame::cameraFrame(cv_bridge::CvImagePtr& cv_img_ptr, cv_bridge::CvImagePtr& cv_depth_ptr, const sensor_msgs::CameraInfoConstPtr& cam_info_ptr)
{
		image_ptr = cv_img_ptr;
		depth_ptr = cv_depth_ptr;
	//TODO: fix this
		fx = 517.3;
		fy = 516.5;
		cx = 318.6;
		cy = 255.3;
}

cameraFrame::~cameraFrame() {
	// TODO Auto-generated destructor stub
}
