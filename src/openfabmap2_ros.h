//
//  openfabmap2_ros.h
//  
//
//  Wrapper by Timothy Morris on 15/04/12.
//

#ifndef _openfabmap2_ros_h
#define _openfabmap2_ros_h

#include "openfabmap.hpp"
#include <openfabmap2/Match.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/CameraInfo.h>
#include "brandwrapper.h"
#include "CDORB.h"
#include "brandwrapper_copy.h"

typedef message_filters::Subscriber<sensor_msgs::Image> image_sub_type;
typedef message_filters::Subscriber<sensor_msgs::CameraInfo> cinfo_sub_type;
//typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
 //                                                       sensor_msgs::Image,
 //                                                       sensor_msgs::CameraInfo> ImagesSyncPolicy;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ImagesSyncPolicy;


namespace openfabmap2_ros 
{	
  enum eDescriptorType {BRAND=0, ORB=1, SURF=2, SIFT=3, CDORB_=4, TEST=5};

  class OpenFABMap2
	{
	public:
		OpenFABMap2(ros::NodeHandle nh);
		
		virtual ~OpenFABMap2();
		
		void subscribeToImages();
		bool isWorking() const;
		
		virtual void shutdown() = 0;
		virtual void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg) = 0;
//		virtual void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg,
//				const sensor_msgs::ImageConstPtr& depth_msg,
//				const sensor_msgs::CameraInfoConstPtr& cam_info_msg) = 0;
		virtual void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg,
				const sensor_msgs::ImageConstPtr& depth_msg) = 0;

	protected:
		ros::NodeHandle nh_;
		
		// Image transport
		image_transport::Subscriber sub_;
		
	    message_filters::Synchronizer<ImagesSyncPolicy>* images_sync_;
	    message_filters::Subscriber<sensor_msgs::Image> *visua_sub_;
	    message_filters::Subscriber<sensor_msgs::Image> *depth_sub_;
	    message_filters::Subscriber<sensor_msgs::CameraInfo> *cinfo_sub_;


		// OpenFABMap2
		of2::FabMap *fabMap;
		cv::Ptr<cv::FeatureDetector> detector;
		cv::Ptr<cv::FeatureDetector> detector2;
		cv::Ptr<cv::DescriptorExtractor>  extractor;
		cv::Ptr<cv::DescriptorMatcher> matcher;
		cv::Ptr<cv::BOWImgDescriptorExtractor> bide;
		std::vector<cv::KeyPoint> kpts;
		
		bool firstFrame_;
		bool visualise_;
		bool working_;
		bool saveQuit_;
		std::string vocabPath_;
		std::string clTreePath_;
		std::string trainbowsPath_;	
		int minDescriptorCount_;
		
		// Data
		cv::Mat vocab;
		cv::Mat clTree;
		cv::Mat trainbows;
		
		double good_matches;
		int counter;
		int num_images;
		std::map<int, int> location_image;
		int last_index;
		double stick;
		double loop_closures;

		eDescriptorType descriptorType;

		double g_min;
		double g_max;

	private:	
		image_transport::ImageTransport it_;
		
		std::string imgTopic_;
		std::string transport_;
	};
	
	//// Running OpenFABMap2
	class FABMapRun : public OpenFABMap2
	{
	public:
		FABMapRun(ros::NodeHandle nh);
		~FABMapRun();
		
		void checkXiSquareMatching();

		void checkDescriptors();

		void computeHomography();

		void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg);
//		void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg,
//				const sensor_msgs::ImageConstPtr& depth_msg,
//				const sensor_msgs::CameraInfoConstPtr& cam_info_msg = 0);
		void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg,
				const sensor_msgs::ImageConstPtr& depth_msg);

		void processImage(cameraFrame& frame);
		void visualiseMatches2(std::vector<of2::IMatch> &matches);
		void visualiseMatches(std::vector<of2::IMatch> &matches);
		bool loadCodebook();
		void shutdown();
		
	private:
		int maxMatches_;
		double minMatchValue_;
		bool disable_self_match_;
		int self_match_window_;
		bool disable_unknown_match_;
		bool only_new_places_;
		
		ros::Publisher pub_;
		std::vector<int> toImgSeq;
		cv::Mat confusionMat;
	};
	
	//// Learning OpenFABMap2
	class FABMapLearn : public OpenFABMap2
	{
	public:
		FABMapLearn(ros::NodeHandle nh);
		~FABMapLearn();
		
		void processImage(cameraFrame& frame);
		void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg);
		void processImgCallback(const sensor_msgs::ImageConstPtr& image_msg,
				const sensor_msgs::ImageConstPtr& depth_msg);
		void findWords();
		void saveCodebook();
		void shutdown();
		
	private:
		int trainCount_;
		int maxImages_;
		double clusterSize_;
		double lowerInformationBound_;
		
		std::vector<cameraFrame> framesSampled;
		
		cv::Mat descriptors;
		cv::Mat bows;
		cv::Ptr<cv::BOWTrainer> trainer;
		of2::ChowLiuTree tree;
	};

}

#endif
