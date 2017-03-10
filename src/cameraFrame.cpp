/*
 * cameraFrame.cpp
 *
 *  Created on: Aug 4, 2016
 *      Author: rasha
 */

#include "openfabmap2/cameraFrame.h"

cameraFrame::cameraFrame() {
	// TODO Auto-generated constructor stub

}

cameraFrame::cameraFrame(cv_bridge::CvImagePtr& cv_ptr)
{
		//image_ptr = cv_ptr;
		color_img = cv_ptr->image;
}

cameraFrame::cameraFrame(cv_bridge::CvImagePtr& cv_img_ptr, cv_bridge::CvImagePtr& cv_depth_ptr, const sensor_msgs::CameraInfoConstPtr& cam_info_ptr)
{
	cv::cvtColor(cv_img_ptr->image, color_img, CV_BGR2RGB);
//	cv_depth_ptr->image.convertTo(depth_img, CV_8UC1, 25.5); //100,0); //TODO: change value
//	depth_img_float = cv_depth_ptr->image;

	valid_depth = (cv_depth_ptr->image == cv_depth_ptr->image);
//	std::cout<<"valid_depth type "<<valid_depth.type()<<std::endl;

	cv_depth_ptr->image.copyTo(depth_img_float, valid_depth);

/*
	image_ptr = cv_img_ptr;
	depth_ptr = cv_depth_ptr;
*/
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

cameraFrame::cameraFrame(cv::Mat& depth_img,  cv::Mat& color_img): depth_img_float(depth_img), color_img(color_img)
{
	fx = 570.34;
	fy = 570.34;
	cx = 319.5;
	cy = 239.5;

}


cameraFrame::~cameraFrame() {
	// TODO Auto-generated destructor stub
}

