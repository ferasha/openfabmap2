/*
 * new_desc.h
 *
 *  Created on: Aug 15, 2016
 *      Author: rasha
 */

#ifndef NEWDESC_H_
#define NEWDESC_H_

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "cameraFrame.h"
#include "scoped_timer.h"

class NewDesc: public cv::DescriptorExtractor {
public:
	NewDesc();
	virtual ~NewDesc();
	void computeImpl( const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints,
	                  cv::Mat& descriptors ) const;
	int descriptorType() const;
	int descriptorSize() const;

	cameraFrame currentFrame;

private:

    int smoothedSum(const cv::Mat& sum, const cv::Point2f& pt) const;
    void compute_orientation(const cv::Mat &image, const cv::Mat& depth_img, std::vector<cv::KeyPoint>& keypoints ) const;
    void getPixelPairs(int index, const cv::KeyPoint& kpt, cv::Point2f& p1, cv::Point2f& p2,
    		float c, float s) const;
    void computeDescriptors(const cv::Mat& gray, const cv::Mat& sum_depth, const std::vector<cv::Mat>& sum_color,
            std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

    cv::Ptr<cv::Feature2D> surf;

    static const int patch_size = 48;
    static const int half_kernel_size = 4;
};

#endif /* NEWDESC_H_ */
