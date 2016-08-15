/*
 * CDORB.h
 *
 *  Created on: Aug 15, 2016
 *      Author: rasha
 */

#ifndef CDORB_H_
#define CDORB_H_

#include <opencv2/features2d/features2d.hpp>
#include "cameraFrame.h"

class CDORB: public cv::DescriptorExtractor {
public:
	CDORB();
	virtual ~CDORB();
	void computeImpl( const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints,
	                  cv::Mat& descriptors ) const;
	int descriptorType() const;
	int descriptorSize() const;

	cameraFrame currentFrame;

private:

    int smoothedSum(const cv::Mat& sum, const cv::Point2f& pt) const;
    void computeAngle(const cv::Mat& gray, cv::KeyPoint& kpt) const;
    void getPixelPairs(int index, cv::Mat& R, const cv::KeyPoint& kpt, float scale, cv::Point2f& p1, cv::Point2f& p2) const;
    void computeDescriptors(const cv::Mat& gray, const cv::Mat& sum_depth, const std::vector<cv::Mat>& sum_color,
            std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

    static const int patch_size = 48;
    static const int half_kernel_size = 4;
};

#endif /* CDORB_H_ */
