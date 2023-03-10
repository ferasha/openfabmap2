/*
 * new_desc.cpp
 *
 *  Created on: Aug 15, 2016
 *      Author: rasha
 */

#include "openfabmap2/new_desc.h"

NewDesc::NewDesc() {
	   cv::initModule_nonfree(); // to use SURF canonical estimation
	   surf = cv::Algorithm::create<cv::Feature2D>("Feature2D.SURF");
	    if( surf.empty() )
	        CV_Error(CV_StsNotImplemented, "OpenCV was built without SURF support.");

}

NewDesc::~NewDesc() {
	// TODO Auto-generated destructor stub
}

int NewDesc::descriptorType() const
{
    return CV_8U;
}

int NewDesc::descriptorSize() const {
	return 32;
}

static const float DEGREE2RAD = (float)CV_PI/180.0;

		   	   	   	   	   	   	   //x1,y1,x2,y2
static int bit_pattern[512 * 4] = { -7,-8,2,-6, -1,5,-8,-20, -15,2,-7,-5,
		   14,18,2,5, 4,-13,14,-18, 11,-19,-2,-13, 10,9,-12,-12, -24,0,16,10,
		   11,-20,-6,-15, -14,-17,22,8, 6,14,1,-23, 16,-4,10,-9, -4,13,20,0, 8,21,-5,3,
		   7,6,3,-23, 9,-21,12,19, -6,-6,-2,0, 6,-18,-4,-11, 16,-12,-1,15, 14,3,-4,16,
		   17,-2,-13,-19, -2,-7,-9,-17, 5,5,5,6, 14,2,7,-6, -4,17,16,-5, 23,-1,7,20,
		   -5,10,-5,-22, -8,6,-5,6, -13,17,9,-18, -20,3,9,9, -13,16,12,-4, -3,9,13,-14,
		   3,-7,3,-5, -12,-9,16,-4, 22,-5,-9,3, 8,-2,18,-7, 10,-19,-7,20, 5,-17,1,-7,
		   0,24,-1,-11, -10,-13,-6,1, -9,8,-4,9, -12,-3,0,11, -16,-8,1,2, 0,-9,2,0,
		   12,-3,-21,3, -16,-16,-13,20, 6,5,9,13, 11,17,-16,7, 8,5,-14,-7, -12,5,8,-2,
		   -8,21,-9,19, -7,-10,-21,-4, 2,-5,-8,5, -16,3,-14,-8, -8,-1,-14,-5,
		   -20,-5,-4,13, 14,17,-16,-16, -14,-8,-7,18, 4,15,11,-13, 19,-11,-12,-4,
		   2,13,4,12, 13,-8,6,14, -5,19,2,-9, 17,-12,22,7, -18,4,10,-20, -6,15,-4,-3,
		   -3,11,-20,-11, 14,9,-9,-7, -9,11,1,18, -6,-20,13,-8, -2,-18,23,4, -21,-5,17,-1,
		   -1,-9,10,1, -2,14,-13,-20, 22,6,-13,13, -14,-14,0,-4, 21,-5,1,-8, -21,0,0,13,
		   -20,-4,8,-20, -6,0,-13,-9, -13,-16,23,2, 14,-8,7,4, 17,16,14,-17, 16,7,21,-10,
		   3,-9,1,10, 1,-22,6,-10, -21,2,12,-17, -18,-8,1,-11, -10,-19,-20,-9, -3,-2,1,8,
		   15,17,6,-15, -11,3,-3,0, 15,8,10,9, -17,4,-1,20, 1,19,-1,-6, 2,-22,-9,-19,
		   -6,17,23,4, 15,-18,-19,-3, -21,10,20,9, 10,17,-9,-6, -2,2,-2,4, -14,17,11,17,
		   -13,-2,-11,5, -13,5,-9,21, 5,2,7,-18, 23,1,-23,-2, -11,8,7,-4, -14,-13,-6,-18,
		   18,11,-1,-7, 12,-6,2,8, 19,-14,9,17, -14,0,-4,12, 11,18,-18,9, -20,-13,0,-21,
		   14,-15,-9,0, 12,-8,-6,15, 2,14,16,7, -20,-1,16,10, 10,20,-8,-16, 2,-10,4,-15,
		   22,-4,16,-12, -18,-12,14,-13, -13,20,14,8, 6,-20,16,5, 20,-11,2,19,
		   8,-15,-16,3, 4,12,-7,5, 3,-17,-5,6, 5,-8,17,6, 2,-8,20,12, 0,9,4,12,
		   14,-4,-4,12, 22,-2,-7,-6, 19,-6,0,6, -11,-18,15,0, -7,-18,-9,16, 0,-19,7,-22,
		   3,12,4,13, 5,-9,1,21, -12,0,19,10, 0,-2,-6,21, 2,7,0,10, -13,9,-14,-17,
		   1,11,-19,11, 1,-8,-23,-4, 16,14,15,-16, 9,5,-6,-4, 13,19,22,-8, 1,-12,-19,-2,
		   -15,-12,-2,4, 21,8,6,-6, -9,12,5,18, -2,-4,-16,2, 21,-7,-15,10, 4,-16,-11,18,
		   22,-8,-4,4, 0,21,-8,-9, 20,2,9,4, -15,-14,12,-11, 6,-13,-18,-12, 23,5,17,-14,
		   -15,4,5,-19, -1,-20,-2,20, 10,-17,12,-18, -6,-11,18,4, -11,-17,-1,-15,
		   17,9,10,-5, -11,6,16,-7, 2,0,-13,-10, 9,-2,-15,-6, -2,-14,-9,-19, 8,15,-4,17,
		   -15,17,20,9, -10,-4,9,-21, 6,-7,-17,-7, 13,7,10,4, 4,-14,-15,11, -20,-3,-9,17,
		   2,8,-17,10, -13,-17,-13,-19, -17,-4,-3,21, 7,-17,0,22, -7,-19,23,1,
		   -19,3,2,-17, 6,-21,20,-6, -2,15,-6,17, 0,7,-5,14, -20,-12,18,-6, -2,-19,-11,13,
		   -21,4,-12,8, 16,-7,-4,-13, -1,15,19,6, 1,12,-14,-5, -20,-7,14,-8,
		   -8,-17,-15,-14, -4,-23,15,-18, -14,-1,11,21, -1,0,-19,-8, -23,4,9,2,
		   8,-16,-4,17, 5,12,-19,3, 16,8,-14,-7, -2,-1,-7,2, -20,1,9,16, -14,-11,3,22,
		   6,21,-14,-1, 8,17,0,-10, -5,8,-14,2, 8,3,21,-10, 11,11,-6,3, -16,17,-8,13,
		   21,-3,-8,-8, 14,-11,19,-14, 0,-4,9,20, 17,15,-12,-20, 9,2,-6,11, 0,14,-8,-17,
		   10,-14,-20,9, 10,-16,3,-8, 15,6,13,2, 23,4,15,16, -9,-22,-18,-15, -10,-6,19,7,
		   -9,21,-1,12, -19,-9,-2,2, -10,-1,14,17, 6,-5,4,17, -11,-8,-1,4, -11,15,-15,10,
		   1,-16,-14,8, -12,19,-4,3, -18,2,-2,-7, 21,-6,10,-20, -9,22,0,19, 12,-7,-5,20,
		   16,9,-6,10, 3,21,-14,-17, 14,8,-18,-10, 2,21,21,-8, 17,5,6,19, -2,-11,-8,6,
		   -23,6,-10,15, -14,7,-2,-21, -12,-19,6,9, 5,15,8,-4, 5,-2,-16,13, 9,-5,-22,5,
		   -6,7,-12,-14, -17,2,-8,-11, -4,-11,15,18, 18,-12,12,-12, 14,-2,3,5,
		   -15,-6,3,-13, -8,14,-6,15, -11,11,0,-18, 7,-3,-7,19, -8,-10,6,12, 15,-10,-5,5,
		   -4,9,16,1, -6,8,15,6, 8,-17,18,3, -17,-14,-14,-13, 3,-16,5,-18, -19,-4,-3,-2,
		   18,7,-17,9, 21,0,-9,4, 8,11,15,13, 16,0,-13,3, -11,-12,8,7, 12,-19,3,19,
		   -12,17,-12,4, 3,9,-4,-15, -18,-11,8,-1, 5,-16,9,2, 0,-12,-12,2, 5,-4,2,19,
		   1,-7,10,-17, -13,2,14,8, 2,-12,4,-19, 8,1,-15,4, 13,12,-20,2, -8,-21,-12,6,
		   -2,20,5,-16, -5,-9,-9,4, -8,-3,5,19, -3,0,-2,-10, -18,8,22,-2, -8,1,-13,4,
		   14,-1,14,8, 8,15,-12,-10, 13,-19,21,0, 2,-19,13,8, -14,16,-7,-10, 16,5,8,10,
		   5,22,-4,-11, 2,9,-10,-5, 8,-14,3,-1, -16,-9,6,-1, 6,-19,-6,-11, -5,23,-17,7,
		   -16,-6,-12,5, 12,9,6,21, 23,-5,-6,-12, 8,-9,-1,5, 19,0,-2,-3, 9,-5,13,12,
		   11,-15,2,9, -1,6,-11,19, 11,21,-8,-16, -6,4,14,-11, 19,-5,8,7, -23,1,7,7,
		   8,-18,-4,-2, 12,15,16,5, -16,-12,-16,11, -8,-21,18,-12, -1,10,-9,-6,
		   -19,-13,-8,2, 7,1,-18,-15, -16,-2,-6,22, -18,-12,-23,1, 0,-22,1,-8, 3,11,19,-6,
		   11,4,-9,-9, 17,12,3,8, -19,-7,11,-13, -1,-14,-5,-5, 14,-6,-18,13, 9,19,20,-1,
		   7,-10,12,15, -15,-4,18,14, 4,-1,7,-16, -18,-7,7,-8, 17,1,12,9, 6,18,-10,-15,
		   -18,9,21,-11, 14,14,0,15, -4,-4,-1,1, 21,11,-18,-12, -4,16,14,12, 7,-15,-1,5,
		   -12,-18,15,0, 23,-3,-8,-3, 20,-3,9,-2, -20,0,12,18, -9,-19,15,4, -12,-6,15,5,
		   18,-1,20,-13, 1,10,7,4, -15,-11,2,-20, 0,-23,-16,12, 7,-6,-7,18, 17,8,-6,1,
		   22,7,5,5, 13,-18,-20,-5, -10,6,9,12, 10,20,-6,17, 7,11,-10,-13, -3,8,-6,-4,
		   -3,-3,0,-10, -9,-15,-14,12, 14,-7,0,13, -2,-15,12,17, -22,2,-7,-6, -6,-2,8,3,
		   15,12,-5,18, -3,22,-10,-13, 23,-6,-15,-4, -5,7,12,13, -1,23,0,-4, 14,-10,4,-16,
		   -8,-14,17,-5, 2,-2,-10,-19, -4,-5,15,0, 9,-9,7,-6, 9,-18,-1,5, 17,-11,-2,9,
		   17,2,7,17, -13,14,16,-7, 13,-11,-3,18, 14,16,-1,13, 12,-2,5,18, -8,-7,15,-8,
		   -7,-12,-6,-18, 19,-10,-6,22, -18,6,8,-14, -2,23,-6,6, -9,18,20,-11,
		   -16,13,-16,-8, 4,-18,15,-7, 7,9,15,-18, -21,11,7,21, -6,9,6,6, 10,-2,2,12,
		   -4,3,7,14, 20,6,0,-24, 7,-11,-6,13, 15,15,-20,-12, -11,17,-3,0, 12,-20,-3,-18,
		   -11,15,14,-2, -5,-14,7,6, -10,20,1,-12, 9,9,3,-21, 0,16,-16,-6, -5,15,-5,0,
		   16,15,-6,16, -1,-23,-5,-22, 19,3,12,-8, -7,-1,0,17, 12,-10,-21,0, -1,13,5,21,
		   9,-13,11,-5, 6,7,-15,-14, -2,-2,5,9, -9,-13,-19,2, -9,14,16,14, 15,-2,-9,0,
		   16,-5,-11,-18, -20,12,19,3, -6,-8,-3,0, -12,-13,-10,9, 11,3,11,19,
		   -17,-12,7,11, -3,-6,11,12, -5,-5,-22,-5, 14,-14,-13,14, -11,10,6,-23,
		   -6,-13,2,7, 12,7,13,11, -18,-14,-6,19, -3,2,9,-14, -1,16,20,13, 7,16,-9,8,
		   11,-7,-4,-2, -4,-3,10,0, 7,17,-18,-1, 6,-4,2,-10, 2,-21,-18,12, 10,4,-14,7,
		   -17,14,0,-24, 11,-11,-3,-10, -10,13,2,8, 9,7,-22,5, -14,9,-8,-2, 14,-17,11,-13,
		   -5,-18,-1,0, 2,16,17,-16, 21,1,-17,10, 12,6,5,4, 10,14,10,-10, 23,0,10,14,
		   10,8,-15,-8, 9,15,1,-9, -15,13,-3,-11, 6,-12,-4,-15, 19,-8,12,9, 12,13,1,14,
		   1,-5,6,-20, -9,13,7,-3, -16,-17,14,-15, -18,5,9,13, 3,15,-21,0, 20,7,-13,-10,
		   -21,-3,15,8, -10,-7,-18,-2, -4,-20,-10,-5, 6,-19,-12,-10, -7,-19,-20,5,
		   4,13,19,-13, -5,-9,-5,15, -9,-10,-15,-3, -14,6,2,11, 10,11,6,-21, 5,-12,9,1,
		   9,-7,-9,6, -6,-10,6,-8, -16,-5,17,-16, -1,7,11,18, 15,8,-11,-18, -19,-2,-14,-6,
		   16,-1,6,-15, 22,2,-2,23, 4,-8,5,-23, -19,-8,-13,-2, 16,-6,-12,12, 3,17,-8,6,
		   -2,3,8,3, 0,5,-4,13, 6,12,5,6, 7,12,21,1, -10,1,10,-2, 4,17,-5,7, -2,9,-2,21,
		   0,-5,-6,-15, 5,4,16,-3, -10,-14,-15,1, 9,-10,6,-18, -2,4,11,3, 11,-1,18,15,
		   3,-1,-19,-2, -16,11,-7,19, -13,-10,0,-8, 6,15,-9,21, 20,3,-7,17, -14,10,-20,-6,
		   4,16,-2,-7, -8,3,21,3, 2,-16,6,3, -4,23,22,-2, -1,4,5,-4, -7,-7,-8,-9 };

void NewDesc::computeImpl( const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints,
                  cv::Mat& descriptors ) const {

	cv::Mat color_Lab;
	cv::cvtColor(currentFrame.color_img, color_Lab, CV_RGB2Lab);

    cv::Mat sum_depth;
    std::vector<cv::Mat> vec_color;
    std::vector<cv::Mat> sum_color(3);

    integral( currentFrame.depth_img_float, sum_depth, CV_64F);
    split(color_Lab, vec_color);
    for (int i=0; i<3; i++) {
    	integral(vec_color[i], sum_color[i], CV_32S);
    }

    cv::KeyPointsFilter::runByImageBorder( keypoints, gray.size(), patch_size/2 + half_kernel_size );
    compute_orientation(gray, currentFrame.depth_img_float, keypoints);
    computeDescriptors(gray, sum_depth, sum_color, keypoints, descriptors);
}

void NewDesc::compute_orientation(const cv::Mat &image,
		const cv::Mat& depth_img, std::vector<cv::KeyPoint>& keypoints ) const
{
	int half_p  = patch_size/2;
	int step_ = (int)currentFrame.depth_img_float.step1();
//	std::cout<<"step "<<step<<std::endl;


   for(int i = 0; i < keypoints.size(); ++i)
   {
/*	   if (i == 8)
	   {
		   std::cout<<"new desc "<<cvRound(keypoints[i].pt.x)<<" "<<cvRound(keypoints[i].pt.y)<<
				   " step "<<step_<<std::endl;
	   }*/
//		const uchar* center_depth = &depth_img.at<uchar> (cvRound(keypoints[i].pt.y), cvRound(keypoints[i].pt.x));
		const float* center_depth_float = &currentFrame.depth_img_float.at<float> (cvRound(keypoints[i].pt.y), cvRound(keypoints[i].pt.x));
//		const uchar* center_depth_valid = &currentFrame.valid_depth.at<uchar> (cvRound(keypoints[i].pt.y), cvRound(keypoints[i].pt.x));

		//double min_depth = std::numeric_limits<double>::max();
	//	uchar min_depth = std::numeric_limits<uchar>::max();
		float min_depth = std::numeric_limits<float>::max();
		for (int v = -half_p; v <= half_p; ++v){
			for (int u = -half_p; u <= half_p; ++u) {
			//	double dp = (double)center_depth[u + v*step_]/25.5;
		//		uchar dp = center_depth[u + v*step_];
				float dp = center_depth_float[u + v*step_];
		//		if (i == 8) {
		//			std::cout<<(double)dp<<":"<<(double)center_depth_valid[u + v*step_]<<" " ;
			//		std::cout<<(double)dp<<":"<<(double)center_depth_valid[u + v*step_]<<" "
			//		<<"("<<v<<","<<u<<","<<v*step_<<","<<u+v*step_<<") " ;
	//			}
				if (dp < min_depth && dp > 0)
					min_depth = dp;
			}
		}
	//	if (i==8)
	//	std::cout<<std::endl<<"min_depth "<<(double)min_depth<<" "<<(double)min_depth/25.5<<std::endl;
   //   keypoints[i].response = std::max( 0.2, (3.8-0.4*std::max(2.0, (double)min_depth/25.5))/3);

		keypoints[i].response = std::max( 0.2, (3.8-0.4*std::max(2.0, (double)min_depth))/3);
	//    keypoints[i].response = std::max( 0.2, (1.2-0.1*std::max(2.0, (double)min_depth)));
	//    keypoints[i].response = std::max( 0.2, (9.8-0.8*std::max(1.0, (double)min_depth))/9);


	//	if (i==8)
    //  std::cout<<"response "<<keypoints[i].response<<std::endl;

      keypoints[i].size = 70.0 * keypoints[i].response;
   }

   surf->operator()(image, cv::noArray(), keypoints, cv::noArray(), true);
}

void NewDesc::getPixelPairs(int index, const cv::KeyPoint& kpt,
		cv::Point2f& p1, cv::Point2f& p2, float c, float s) const {

    p1.x = bit_pattern[index * 4]     * kpt.response;
    p1.y = bit_pattern[index * 4 + 1] * kpt.response;

    p2.x = bit_pattern[index * 4 + 2] * kpt.response;
    p2.y = bit_pattern[index * 4 + 3] * kpt.response;

    float temp = p1.x;
    p1.x = c*p1.x - s*p1.y;
    p1.y = s*temp + c*p1.y;
    temp = p2.x;
    p2.x = c*p2.x - s*p2.y;
    p2.y = s*temp + c*p2.y;

    p1.x += (int)(kpt.pt.x + 0.5);
    p1.y += (int)(kpt.pt.y + 0.5);
    p2.x += (int)(kpt.pt.x + 0.5);
    p2.y += (int)(kpt.pt.y + 0.5);

}

inline int NewDesc::smoothedSum(const cv::Mat& sum, const cv::Point2f& pt) const
{
/*
	const int* center = &sum.at<int>(pt.y, pt.x);
	int step_ = (int)sum.step1();

    return   center[step_*(half_kernel_size + 1) + half_kernel_size + 1]
           - center[step_*(half_kernel_size + 1) - half_kernel_size]
           - center[step_*(-half_kernel_size) + half_kernel_size + 1]
           + center[step_*(-half_kernel_size) - half_kernel_size];
*/

    return   sum.at<int>(pt.y + half_kernel_size + 1, pt.x + half_kernel_size + 1)
           - sum.at<int>(pt.y + half_kernel_size + 1, pt.x - half_kernel_size)
           - sum.at<int>(pt.y - half_kernel_size,     pt.x + half_kernel_size + 1)
           + sum.at<int>(pt.y - half_kernel_size,     pt.x - half_kernel_size);

}

inline double NewDesc::smoothedSumDepth(const cv::Mat& sum, const cv::Point2f& pt) const
{
    return   sum.at<double>(pt.y + half_kernel_size + 1, pt.x + half_kernel_size + 1)
           - sum.at<double>(pt.y + half_kernel_size + 1, pt.x - half_kernel_size)
           - sum.at<double>(pt.y - half_kernel_size,     pt.x + half_kernel_size + 1)
           + sum.at<double>(pt.y - half_kernel_size,     pt.x - half_kernel_size);

}
void NewDesc::computeDescriptors(const cv::Mat& gray, const cv::Mat& sum_depth, const std::vector<cv::Mat>& sum_color,
        std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {

	descriptors = cv::Mat::zeros((int)keypoints.size(), 32, CV_8U);

    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        uchar* cdesc = descriptors.ptr(i);
        cv::KeyPoint& kpt = keypoints[i];
        float angle = kpt.angle * DEGREE2RAD;

        float cos_ang = cos(angle);
        float sin_ang = sin(angle);

        int c1, c2;
        double d1, d2;
        int step = 8;
        for( int j = 0; j < step; j++ )
        {
        	int index = j*8;
        	for (int k = 0; k < 8; k++, index++)
            {
            	cv::Point2f p1, p2;

        		getPixelPairs(index, kpt, p1, p2, cos_ang, sin_ang);
        		c1 = smoothedSum( sum_color[0], p1 );
        		c2 = smoothedSum( sum_color[0], p2 );
        		cdesc[j] += (c1 < c2) << (7-k);

        		getPixelPairs(index+step*8, kpt, p1, p2, cos_ang, sin_ang);
        		c1 = smoothedSum( sum_color[1], p1 );
        		c2 = smoothedSum( sum_color[1], p2 );
        		cdesc[j+step] += (c1 < c2) << (7-k);

        		getPixelPairs(index+step*8*2, kpt, p1, p2, cos_ang, sin_ang);
        		c1 = smoothedSum( sum_color[2], p1 );
        		c2 = smoothedSum( sum_color[2], p2 );
        		cdesc[j+step*2] += (c1 < c2) << (7-k);

        		getPixelPairs(index+step*8*3, kpt, p1, p2, cos_ang, sin_ang);
        		d1 = smoothedSumDepth( sum_depth, p1 );
        		d2 = smoothedSumDepth( sum_depth, p2 );
        		cdesc[j+step*3] += (d1 < d2) << (7-k);
            }
        }
    }
}


