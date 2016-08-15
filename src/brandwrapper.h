/*
 * brandwrapper.h
 *
 *  Created on: Aug 3, 2016
 *      Author: rasha
 */

#ifndef BRANDWRAPPER_H_
#define BRANDWRAPPER_H_

#include "brand.h"
#include "cameraFrame.h"


class brand_wrapper: public cv::DescriptorExtractor {
public:
	brand_wrapper();
	virtual ~brand_wrapper();
	void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
	                  cv::Mat& descriptors ) const;
	int descriptorType() const;
	int descriptorSize() const;

//	void updateData(cv_bridge::CvImagePtr cv_image_ptr, cv_bridge::CvImagePtr cv_depth_ptr, const sensor_msgs::CameraInfo& cam_info) const;

	cameraFrame currentFrame;
private:
	void compute_normals(const cv::Mat& cloud, cv::Mat& normals, cv::Mat& angles) const;
	void create_cloud( const cv::Mat &depth,
	                   float fx, float fy, float cx, float cy,
	                   cv::Mat& cloud ) const;

	BrandDescriptorExtractor brand;
};


#endif /* BRANDWRAPPER_H_ */
