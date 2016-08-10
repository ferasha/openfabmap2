/*
 * brandwrapper.cpp
 *
 *  Created on: Aug 3, 2016
 *      Author: rasha
 */

#include "brandwrapper.h"
#include <pcl/features/integral_image_normal.h>

brand_wrapper::brand_wrapper() {
	// TODO Auto-generated constructor stub

}

brand_wrapper::~brand_wrapper() {
	// TODO Auto-generated destructor stub
}

void brand_wrapper::computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                  cv::Mat& descriptors ) const {

	cv::Mat cloud, normals;
	create_cloud(currentFrame.depth_ptr->image, currentFrame.fx, currentFrame.fy, currentFrame.cx, currentFrame.cy, cloud );
//	compute_normals( cloud, normals );
	brand.compute(image, cloud, normals, keypoints, descriptors);

}

/*
void brand_wrapper::updateData(cv_bridge::CvImagePtr cv_image_ptr, cv_bridge::CvImagePtr cv_depth_ptr, const sensor_msgs::CameraInfo& cam_info) const {
	setCameraIntrinsics(cam_info);
	currentFrame.image_ptr = cv_image_ptr;
	currentFrame.depth_ptr = cv_depth_ptr;
}
*/

int brand_wrapper::descriptorType() const
{
    return CV_8U;
}

int brand_wrapper::descriptorSize() const {
	return brand.getDescriptorSize();
}

void brand_wrapper::compute_normals(const cv::Mat& cloud, cv::Mat& normals) const
{
   pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud( new pcl::PointCloud<pcl::PointXYZ> );

   pcl_cloud->clear();
   pcl_cloud->width     = cloud.cols;
   pcl_cloud->height    = cloud.rows;
   pcl_cloud->points.resize( pcl_cloud->width * pcl_cloud->height);

   for(int y = 0; y < cloud.rows; ++y)
   for(int x = 0; x < cloud.cols; ++x)
   {
      pcl_cloud->at(x,y).x = cloud.at<cv::Point3f>(y,x).x;
      pcl_cloud->at(x,y).y = cloud.at<cv::Point3f>(y,x).y;
      pcl_cloud->at(x,y).z = cloud.at<cv::Point3f>(y,x).z;
   }

   pcl::PointCloud<pcl::Normal>::Ptr pcl_normals (new pcl::PointCloud<pcl::Normal>);
   pcl_normals->clear();
   pcl_normals->width  = pcl_cloud->width;
   pcl_normals->height = pcl_cloud->height;
   pcl_normals->points.resize(pcl_cloud->width * pcl_cloud->height);

   pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
   ne.setInputCloud( pcl_cloud );

   ne.setNormalSmoothingSize( 5 );
   ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
   ne.compute( *pcl_normals );

   normals.create( cloud.size(), CV_32FC3 );

   for(int y = 0; y < pcl_normals->height; ++y)
   for(int x = 0; x < pcl_normals->width; ++x)
   {
      normals.at<cv::Point3f>(y,x).x = pcl_normals->at(x,y).normal_x;
      normals.at<cv::Point3f>(y,x).y = pcl_normals->at(x,y).normal_y;
      normals.at<cv::Point3f>(y,x).z = pcl_normals->at(x,y).normal_z;
   }
}

void brand_wrapper::create_cloud( const cv::Mat &depth,
                   float fx, float fy, float cx, float cy,
                   cv::Mat& cloud ) const
{
    const float inv_fx = 1.f/fx;
    const float inv_fy = 1.f/fy;

    cloud.create( depth.size(), CV_32FC3 );

    for( int y = 0; y < cloud.rows; y++ )
    {
        cv::Point3f* cloud_ptr = (cv::Point3f*)cloud.ptr(y);
        const uint16_t* depth_prt = (uint16_t*)depth.ptr(y);

        for( int x = 0; x < cloud.cols; x++ )
        {
            float d = (float)depth_prt[x]/1000; // meters
            cloud_ptr[x].x = (x - cx) * d * inv_fx;
            cloud_ptr[x].y = (y - cy) * d * inv_fy;
            cloud_ptr[x].z = d;
        }
    }
}


