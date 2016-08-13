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

	/*  // TUM?
		fx = 517.3;
		fy = 516.5;
		cx = 318.6;
		cy = 255.3;
	*/
		//nao
		fx = 570.34;
		fy = 570.34;
		cx = 319.5;
		cy = 239.5;

}

cameraFrame::cameraFrame(cv::Mat& depth_img): depth_img(depth_img)
{
	fx = 570.34;
	fy = 570.34;
	cx = 319.5;
	cy = 239.5;

}


cameraFrame::~cameraFrame() {
	// TODO Auto-generated destructor stub
}

