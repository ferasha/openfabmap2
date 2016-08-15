/* 
   Copyright (C) 2012-2013 Erickson R. Nascimento

   THIS SOURCE CODE IS PROVIDED 'AS-IS', WITHOUT ANY EXPRESS OR IMPLIED
   WARRANTY. IN NO EVENT WILL THE AUTHOR BE HELD LIABLE FOR ANY DAMAGES
   ARISING FROM THE USE OF THIS SOFTWARE.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:


   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   Contact: erickson [at] dcc [dot] ufmg [dot] br

*/

#ifndef BRAND_copy_H_
#define BRAND_copy_H_

#include <vector>
#include <opencv2/features2d/features2d.hpp>

class BrandDescriptorExtractor_copy
{

public:
   // - degree_threshold in degrees
   // - descriptor_size  in bytes: 16, 32 or 64
    BrandDescriptorExtractor_copy( double degree_threshold = 45, int desc_size = 32 );

    int getDescriptorSize() const;

   // - image: grayscale
   // - cloud: matrix of cv::Point3f (3D points)
   // - normals: matrix of cv::Point3f (3D points)
   //
   // - pixels in image, points in cloud and normals must be related, e.g, pixel
   //    image.at<uchar>(x,y) is represented by point cloud.at<Point3f>(x,y) which
   //    has normal normals.at<Point3f>(x,y)
    void compute(	  const cv::Mat &image,
    				const cv::Mat& color,
    				const cv::Mat& depth,
                    const cv::Mat& cloud,
                    const cv::Mat& normals,
                    const cv::Mat& angles,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::Mat& descriptors ) const;
private:

   void extract_features(  const cv::Mat& cloud,
                           const cv::Mat& normals,
                           const cv::Mat& angles,
                           const cv::Mat &image,
                           const cv::Mat& color,
                           const cv::Mat& depth,
                           std::vector<cv::KeyPoint>& keypoints, 
                           cv::Mat& intensity, cv::Mat& shape, cv::Mat& color_desc ) const;

   void canonical_orientation(  const cv::Mat& img, const cv::Mat& mask,
                                std::vector<cv::KeyPoint>& keypoints ) const;

    void compute_intensity_descriptors(const cv::Mat& image,
    									         std::vector<cv::KeyPoint>& keypoints,
    									         cv::Mat& descriptors );

    void compute_intensity_and_shape_descriptors(   const cv::Mat& image,
    												const cv::Mat& color,
    												const cv::Mat& depth,
                                                    const cv::Mat& cloud,
                                                    const cv::Mat& normals,
                                                    const cv::Mat& angles,
                                                    std::vector<cv::KeyPoint>& keypoints,
                                                    cv::Mat& idescriptors,
                                                    cv::Mat& sdescriptors,
                                                    cv::Mat& cdescriptors) const;

    int smoothedSum(const cv::Mat& sum, const cv::Point2f& pt) const;
    float smoothedSumAngle(const cv::Mat& sum, const cv::Point2f& pt) const;

    void computeAngle(const cv::Mat& image, const cv::Mat& depth_img, std::vector<cv::KeyPoint>& kpts) const;

    void pixelTests(const cv::Mat& sum,
    				const cv::Mat& sum_depth,
    				const std::vector<cv::Mat>& sum_color,
    				const cv::Mat& sum_angle,
                    const cv::Mat& cloud,
                    const cv::Mat& normals,
                    const std::vector<cv::KeyPoint>& keypoints, 
                    cv::Mat& idescriptors, cv::Mat& sdescriptors, cv::Mat& cdescriptors ) const;

    int      m_descriptor_size;
    double   m_degree_threshold;

    static const int m_patch_size = 24;
    static const int m_half_kernel_size = 4;
};

#endif 
