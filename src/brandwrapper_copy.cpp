/*
 * brandwrapper.cpp
 *
 *  Created on: Aug 3, 2016
 *      Author: rasha
 */

#include "openfabmap2/brandwrapper_copy.h"
#include <pcl/features/integral_image_normal.h>

brand_wrapper_copy::brand_wrapper_copy() {
	// TODO Auto-generated constructor stub

}

brand_wrapper_copy::~brand_wrapper_copy() {
	// TODO Auto-generated destructor stub
}

void brand_wrapper_copy::computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                  cv::Mat& descriptors ) const {

	cv::Mat cloud, normals, angles;
//	create_cloud(currentFrame.depth_ptr->image, currentFrame.fx, currentFrame.fy, currentFrame.cx, currentFrame.cy, cloud );
	create_cloud(currentFrame.depth_img, currentFrame.fx, currentFrame.fy, currentFrame.cx, currentFrame.cy, cloud );
	compute_normals( cloud, normals, angles );
//	cv::Mat brand_desc;

	brand.compute(image, currentFrame.color_img, currentFrame.depth_img, cloud, normals,
			angles, keypoints, descriptors);
/*
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::OrbDescriptorExtractor();
	cv::Mat orb_desc;
	extractor->compute(image, keypoints, orb_desc);
	hconcat(brand_desc, orb_desc, descriptors);
*/
}

/*
void brand_wrapper_copy::updateData(cv_bridge::CvImagePtr cv_image_ptr, cv_bridge::CvImagePtr cv_depth_ptr, const sensor_msgs::CameraInfo& cam_info) const {
	setCameraIntrinsics(cam_info);
	currentFrame.image_ptr = cv_image_ptr;
	currentFrame.depth_ptr = cv_depth_ptr;
}
*/

int brand_wrapper_copy::descriptorType() const
{
    return CV_8U;
}

int brand_wrapper_copy::descriptorSize() const {
	return brand.getDescriptorSize();
}

void brand_wrapper_copy::compute_normals(const cv::Mat& cloud, cv::Mat& normals, cv::Mat& angles) const
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
/*
   ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
   ne.setMaxDepthChangeFactor(0.02f);
   ne.setNormalSmoothingSize(10.0f);
 */
   ne.compute( *pcl_normals );

   normals.create( cloud.size(), CV_32FC3 );
   angles.create( cloud.size(), CV_32FC1 );

   cv::Point3f vertical(0,1,0);
   int count = 0;
   for(int y = 0; y < pcl_normals->height; ++y) {
	   for(int x = 0; x < pcl_normals->width; ++x)
	   {
		  cv::Point3f& normal = normals.at<cv::Point3f>(y,x);
		  normal.x = pcl_normals->at(x,y).normal_x;
		  normal.y = pcl_normals->at(x,y).normal_y;
		  normal.z = pcl_normals->at(x,y).normal_z;
		  angles.at<float>(y,x) = normal.dot(vertical) / cv::norm(vertical);
		  if (isnan(angles.at<float>(y,x))) {
			  angles.at<float>(y,x) = 0;
			 // std::cout<<"angle is nan"<<std::endl;
			  count += 1;
		  }
	   }
   }
//   std::cout<<"count angle nan "<<count<<std::endl;
}

void brand_wrapper_copy::create_cloud( const cv::Mat &depth,
                   float fx, float fy, float cx, float cy,
                   cv::Mat& cloud ) const
{
    const float inv_fx = 1.f/fx;
    const float inv_fy = 1.f/fy;

 //   std::cout<<"mat depth type "<<depth.type()<<std::endl;

    cloud.create( depth.size(), CV_32FC3 );

    for( int y = 0; y < cloud.rows; y++ )
    {
        cv::Point3f* cloud_ptr = (cv::Point3f*)cloud.ptr(y);
  //      const uint16_t* depth_prt = (uint16_t*)depth.ptr(y);
        const uchar* depth_prt = (uchar*)depth.ptr(y);

        for( int x = 0; x < cloud.cols; x++ )
        {
            float d = (float)depth_prt[x]/25.5; // meters
            cloud_ptr[x].x = (x - cx) * d * inv_fx;
            cloud_ptr[x].y = (y - cy) * d * inv_fy;
            cloud_ptr[x].z = d;
            if (isnan(d))
            	std::cout<<"(nan)";
    //        std::cout<<d<<" ";
        }
   //     std::cout<<std::endl;
    }
}


