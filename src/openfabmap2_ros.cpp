//
//  openfabmap2_ros.cpp
//  
//
//  Wrapper by Timothy Morris on 15/04/12.
//

#include "openfabmap2/openfabmap2_ros.h"
#include <iostream>
#include <openfabmap2/Match.h>
#include <ros/console.h>
#include <sstream>
#include <fstream>
#include <string>

namespace enc = sensor_msgs::image_encodings;

namespace openfabmap2_ros 
{
  /////////////////////////////////
	//// *** OpenFABMap2 ROS BASE ***
	/////////////////////////////////
	//// Constructor
	// Pre:
	// Post: --Load parameters
	OpenFABMap2::OpenFABMap2(ros::NodeHandle nh) : 
	nh_(nh), it_(nh),
	firstFrame_(true), visualise_(false), working_(true), saveQuit_(false)
	{
		// TODO: finish implementing parameter server
		// Read private parameters
		ros::NodeHandle local_nh_("~");
		local_nh_.param<std::string>("vocab", vocabPath_, "vocab.yml");
		local_nh_.param<std::string>("clTree", clTreePath_, "clTree.yml");
		local_nh_.param<std::string>("trainbows", trainbowsPath_, "trainbows.yml");
		local_nh_.param<std::string>("transport", transport_, "raw");
		local_nh_.param<bool>("visualise", visualise_, false);
		local_nh_.param<int>("MinDescriptorCount", minDescriptorCount_, 50);
		
/*
		if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) ) {
		   ros::console::notifyLoggerLevelsChanged();
		}
*/
		// Read node parameters
		imgTopic_ = nh_.resolveName("image");
		
		// Initialise feature method

		//////////
		// Surf parameters that may be used for both 'detector' and 'extractor'
		int surf_hessian_threshold, surf_num_octaves, surf_num_octave_layers, surf_upright, surf_extended;
		
		local_nh_.param<int>("HessianThreshold", surf_hessian_threshold, 1000);
		local_nh_.param<int>("NumOctaves", surf_num_octaves, 4);
		local_nh_.param<int>("NumOctaveLayers", surf_num_octave_layers, 2);
		local_nh_.param<int>("Extended", surf_extended, 0);
		local_nh_.param<int>("Upright", surf_upright, 1);
		
		//////////
		//create common feature detector
		std::string detectorType;
		local_nh_.param<std::string>("DetectorType", detectorType, "FAST");

		if(detectorType == "STAR") {	
			int star_max_size, star_response, star_line_threshold, star_line_binarized, star_suppression;
			local_nh_.param<int>("MaxSize", star_max_size, 32);
			local_nh_.param<int>("Response", star_response, 10);
			local_nh_.param<int>("LineThreshold", star_line_threshold, 18);
			local_nh_.param<int>("LineBinarized", star_line_binarized, 18);
			local_nh_.param<int>("Suppression", star_suppression, 20);			
			detector = new cv::StarFeatureDetector(star_max_size,
																						 star_response, 
																						 star_line_threshold,
																						 star_line_binarized,
																						 star_suppression);
			
		} else if(detectorType == "FAST") {
			int fast_threshold, fast_non_max_suppression;
			local_nh_.param<int>("Threshold", fast_threshold, 50);
			local_nh_.param<int>("NonMaxSuppression", fast_non_max_suppression, 1);													 
			detector = new cv::FastFeatureDetector(fast_threshold,
																						 fast_non_max_suppression > 0);
			
		} else if(detectorType == "SURF") {
			detector = new cv::SURF(surf_hessian_threshold, 
																						 surf_num_octaves, 
																						 surf_num_octave_layers, 
																						 surf_extended > 0,
																						 surf_upright > 0);
			detector2 = new cv::SURF(surf_hessian_threshold,
																						 surf_num_octaves,
																						 surf_num_octave_layers,
																						 surf_extended > 0,
																						 surf_upright > 0);
			
		} else if(detectorType == "SIFT") {
			int sift_nfeatures, sift_num_octave_layers;
			double sift_threshold, sift_edge_threshold, sift_sigma;			
			local_nh_.param<int>("NumFeatures", sift_nfeatures, 0);
			local_nh_.param<int>("NumOctaveLayers", sift_num_octave_layers, 3);
			local_nh_.param<double>("Threshold", sift_threshold, 0.04);
			local_nh_.param<double>("EdgeThreshold", sift_edge_threshold, 10);
			local_nh_.param<double>("Sigma", sift_sigma, 1.6);
			detector = new cv::SIFT(500,
															sift_num_octave_layers,
															sift_threshold,
															sift_edge_threshold,
															sift_sigma);
			detector2 = new cv::SIFT(500,
															sift_num_octave_layers,
															sift_threshold,
															sift_edge_threshold,
															sift_sigma);

			
		} else if(detectorType == "ORB") {
			detector = new cv::OrbFeatureDetector(500);
			detector2 = new cv::OrbFeatureDetector(500);

		} else {
			int mser_delta, mser_min_area, mser_max_area, mser_max_evolution, mser_edge_blur_size;
			double mser_max_variation, mser_min_diversity, mser_area_threshold, mser_min_margin;
			local_nh_.param<int>("Delta", mser_delta, 5);
			local_nh_.param<int>("MinArea", mser_min_area, 60);
			local_nh_.param<int>("MaxArea", mser_max_area, 14400);
			local_nh_.param<double>("MaxVariation", mser_max_variation, 0.25);
			local_nh_.param<double>("MinDiversity", mser_min_diversity, 0.2);
			local_nh_.param<int>("MaxEvolution", mser_max_evolution, 200);
			local_nh_.param<double>("AreaThreshold", mser_area_threshold, 1.01);
			local_nh_.param<double>("MinMargin", mser_min_margin, 0.003);
			local_nh_.param<int>("EdgeBlurSize", mser_edge_blur_size, 5);
			detector = new cv::MSER(mser_delta,
																						 mser_min_area,
																						 mser_max_area,
																						 mser_max_variation,
																						 mser_min_diversity,
																						 mser_max_evolution,
																						 mser_area_threshold,
																						 mser_min_margin,
																						 mser_edge_blur_size);
		}
		
		std::string descType;
		local_nh_.param<std::string>("DescriptorType", descType, "ORB");

		if (descType == "ORB")
		{
			descriptorType = ORB;
			extractor = new cv::OrbDescriptorExtractor();
			matcher = new cv::BFMatcher(cv::NORM_HAMMING);
		}
		else if (descType == "CDORB")
		{
			descriptorType = CDORB_;
			extractor = new CDORB();
			matcher = new cv::BFMatcher(cv::NORM_HAMMING);
		}
		else if (descType == "TEST")
		{
			descriptorType = TEST;
			extractor = new NewDesc();
			matcher = new cv::BFMatcher(cv::NORM_HAMMING);
		}
		else if (descType == "BRAND")
		{
			descriptorType = BRAND;
			extractor = new brand_wrapper();
			matcher = new cv::BFMatcher(cv::NORM_HAMMING);
		}
		else if (descType == "SIFT") {
			descriptorType = SIFT;
			extractor = new cv::SIFT();
		//	matcher = new cv::FlannBasedMatcher();
			matcher = new cv::BFMatcher(cv::NORM_L1);
		}
		else
		{
			descriptorType = SURF;
/*			extractor = new cv::SURF(surf_hessian_threshold,
															 surf_num_octaves,
																									surf_num_octave_layers,
																									surf_extended > 0,
																									surf_upright > 0);
*/
//			extractor = new cv::SURF(surf_hessian_threshold, surf_num_octaves, surf_num_octave_layers, surf_extended > 0, true);


			extractor = new cv::SURF();
		//	matcher = new cv::FlannBasedMatcher();
			matcher = new cv::BFMatcher(cv::NORM_L1);
		}

		bide = new cv::BOWImgDescriptorExtractor(extractor, matcher);
	}
	
	// Destructor
	OpenFABMap2::~OpenFABMap2()
	{
	}
	
	//// Set Callback
	// Pre: --Valid 'imgTopic_' exists
	// Post: --Subscribes for Images with 'processImgCallback'
	void OpenFABMap2::subscribeToImages()
	{
		// Subscribe to images
		ROS_INFO("Subscribing to:\n\t* %s", 
						 imgTopic_.c_str());
		int q = 100000;
		
		num_images = 0;

		g_min = std::numeric_limits<double>::max();
		g_max = std::numeric_limits<double>::min();


//		if (descriptorType != BRAND)
//		sub_ = it_.subscribe(imgTopic_, q, &OpenFABMap2::processImgCallback, this, transport_);

//		else {
			visua_sub_ = new image_sub_type(nh_, "/camera/rgb/image_color", q);
			depth_sub_ = new image_sub_type(nh_, "/camera/depth/image", q);
		//	visua_sub_ = new image_sub_type(nh_, "/camera/rgb/image_raw", q);
		//	depth_sub_ = new image_sub_type(nh_, "/camera/depth_registered/image_raw", q);
//			cinfo_sub_ = new cinfo_sub_type(nh_, "/camera/rgb/camera_info", q);
//			images_sync_ = new message_filters::Synchronizer<ImagesSyncPolicy>(ImagesSyncPolicy(q),  *visua_sub_, *depth_sub_, *cinfo_sub_);
//			images_sync_->registerCallback(boost::bind(&OpenFABMap2::processImgCallback, this, _1, _2, _3));
			images_sync_ = new message_filters::Synchronizer<ImagesSyncPolicy>(ImagesSyncPolicy(q),  *visua_sub_, *depth_sub_);
			images_sync_->registerCallback(boost::bind(&OpenFABMap2::processImgCallback, this, _1, _2));

//		}

	}
	
	//// Running Check
	// Pre: none
	// Post: none
	bool OpenFABMap2::isWorking() const
	{
		return working_;
	}
	// end class implemtation OpenFABMap2

	//////////////////
	//// *** LEARN ***
	//////////////////
	//// Constructor
	// Pre: Valid NodeHandle provided
	// Post: --Calls 'subscribeToImages'
	FABMapLearn::FABMapLearn(ros::NodeHandle nh) : 
	OpenFABMap2(nh), trainCount_(0)
	{
		// Read private parameters
		ros::NodeHandle local_nh_("~");
		local_nh_.param<int>("maxImages", maxImages_, 10);
		local_nh_.param<double>("clusterSize", clusterSize_, 0.6);
		local_nh_.param<double>("LowerInformationBound", lowerInformationBound_, 0);

        cv::TermCriteria terminate_criterion;
        terminate_criterion.epsilon = FLT_EPSILON;
		if (descriptorType == BRAND || descriptorType == ORB || descriptorType == CDORB_ || descriptorType == TEST) {
			trainer = new of2::BoWKmeansppBinaryTrainer(2000, terminate_criterion, 1, cv::KMEANS_PP_CENTERS );
		}
		else
			trainer = new cv::BOWKMeansTrainer(2000, terminate_criterion, 1, cv::KMEANS_PP_CENTERS );
//			trainer = new of2::BOWMSCTrainer(clusterSize_);
		
		subscribeToImages();
	}
	
	//// Destructor
	FABMapLearn::~FABMapLearn()
	{
	}
	
	//// Image Callback
	// Pre: 
	// Post: --Calls 'shutdown'
	void FABMapLearn::processImgCallback(const sensor_msgs::ImageConstPtr& image_msg)
	{
		ROS_DEBUG_STREAM("Learning image sequence number: " << image_msg->header.seq);
		cv_bridge::CvImagePtr cv_ptr;
		try
		{
			cv_ptr = cv_bridge::toCvCopy(image_msg, enc::MONO8);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

		cameraFrame frame(cv_ptr);
		processImage(frame);
	}

	void FABMapLearn::processImgCallback(const sensor_msgs::ImageConstPtr& image_msg,
			const sensor_msgs::ImageConstPtr& depth_msg)
	{
		num_images += 1;
//		std::cout<<"num images "<<num_images<<std::endl;
//		return;


		ROS_DEBUG_STREAM("Learning image sequence number: " << image_msg->header.seq);
		cv_bridge::CvImagePtr cv_ptr;
		cv_bridge::CvImagePtr cv_depth_ptr;
		try
		{
			cv_ptr = cv_bridge::toCvCopy(image_msg); //, enc::MONO8);
			cv_depth_ptr = cv_bridge::toCvCopy(depth_msg);//, enc::MONO8);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

		sensor_msgs::CameraInfoConstPtr cam_info_msg;

		cameraFrame frame(cv_ptr, cv_depth_ptr, cam_info_msg);

		if (descriptorType == BRAND)
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
		else if (descriptorType == CDORB_)
			static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
		else if (descriptorType == TEST)
			static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = frame;

		processImage(frame);
	}

void FABMapLearn::processImage(cameraFrame& currentFrame) {

		ROS_DEBUG("Received %d by %d image, depth %d, channels %d", currentFrame.color_img.cols,currentFrame.color_img.rows,
				currentFrame.color_img.depth(), currentFrame.color_img.channels());
		
		ROS_DEBUG("--Detect");
		detector->detect(currentFrame.color_img, kpts);
		ROS_DEBUG("--Extract");
		extractor->compute(currentFrame.color_img, kpts, descriptors);
		
//		std::cout<<descriptors.row(0)<<std::endl;

		// Check if frame was useful
		if (!descriptors.empty() && kpts.size() > minDescriptorCount_)
		{
		/*
			cv::Mat centre1 = cv::Mat::zeros(1,descriptors.cols,descriptors.type());
			cv::Mat centre2 = cv::Mat::zeros(1,descriptors.cols,descriptors.type());
			centre2.setTo(255);
			cv::Mat centre3 = cv::Mat::zeros(1,descriptors.cols,descriptors.type());
			centre3.setTo(7);
			trainer->add(centre1);
			trainer->add(centre2);
			trainer->add(centre3);
		 */
			trainer->add(descriptors);

			trainCount_++;
			ROS_INFO_STREAM("--Added to trainer" << " (" << trainCount_ << " / " << maxImages_ << ")");
			
			// Add the frame to the sample pile
			// cv_bridge::CvImagePtr are smart pointers
			framesSampled.push_back(currentFrame);
			
			if (visualise_)
			{
				ROS_DEBUG("Attempting to visualise key points.");
				cv::Mat feats;
				cv::drawKeypoints(currentFrame.color_img, kpts, feats);
				
				cv::imshow("KeyPoints", feats);
				char c = cv::waitKey(10);
				// TODO: nicer exit
				if(c == 27) 
				{
					working_ = false;
					saveQuit_ = true;
				}
			}
		}
		else
		{
			ROS_WARN("--Image not descriptive enough, ignoring.");
		}
		
		// TODO: cv::waitKey(10) // Console triggered save&close
		if ((!(trainCount_ < maxImages_) && maxImages_ > 0) || saveQuit_)
		{
			shutdown();			
		}
	}
	
	//// Find words
	// Pre: 
	// Post:
	void FABMapLearn::findWords()
	{
		cv::Mat bow;
		
		for (std::vector<cameraFrame>::iterator frameIter = framesSampled.begin();
				 frameIter != framesSampled.end();
				 ++frameIter)
		{
			detector->detect((*frameIter).color_img, kpts);
			if (descriptorType == BRAND)
				static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = (*frameIter);
			else if (descriptorType == CDORB_)
				static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = (*frameIter);
			else if (descriptorType == TEST)
				static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = (*frameIter);

			bide->compute((*frameIter).color_img, kpts, bow);
			bows.push_back(bow);
		}
	}
	
	//// File saver
	// Pre: Application has write premission to path provided
	// Post: YML files are written for 'vocab' 'clTree' and 'bows'
	void FABMapLearn::saveCodebook()
	{
		ROS_INFO("Saving codebook...");
		cv::FileStorage fs;
		
		ROS_INFO_STREAM("--Saving Vocabulary to " << vocabPath_);		
		fs.open(vocabPath_,
		  			cv::FileStorage::WRITE);
		fs << "Vocabulary" << vocab;
		fs.release();
		
		ROS_INFO_STREAM("--Saving Chow Liu Tree to " << clTreePath_);
		fs.open(clTreePath_,
		 				cv::FileStorage::WRITE);
		fs << "Tree" << clTree;
		fs.release();
		
		ROS_INFO_STREAM("--Saving Trained Bag of Words to " << trainbowsPath_);
		fs.open(trainbowsPath_,
		 				cv::FileStorage::WRITE);
		fs << "Trainbows" << bows;
		fs.release();
	}
	
	//// Unlink Callback
	// Pre:
	// Post: --Calls 'saveCodebook' --Cleanup
	void FABMapLearn::shutdown()
	{
		ROS_INFO("Clustering to produce vocabulary");
		vocab = trainer->cluster();
		ROS_INFO("Vocabulary contains %d words, %d dims",vocab.rows,vocab.cols);
		
		ROS_INFO("Setting vocabulary...");
		bide->setVocabulary(vocab);
		
		ROS_INFO("Gathering BoW's...");
		findWords();
		
		ROS_INFO("Making the Chow Liu tree...");
		tree.add(bows);
		clTree = tree.make(lowerInformationBound_);
		
		ROS_INFO("Saving work completed...");
		saveCodebook();
		
		// Flag this worker as complete
		working_ = false;
		
		nh_.shutdown();

		if (sub_.getNumPublishers() > 0)
		{
			// Un-subscribe to Images
			ROS_WARN_STREAM("Shutting down " << sub_.getNumPublishers() << " subscriptions...");
			sub_.shutdown();
			nh_.shutdown();
		}
		else
		{
			ROS_ERROR("Shutdown called with no existing subscriptions...");
		}
	}
	// end class implementation FABMapLearn

	////////////////
	//// *** RUN ***
	////////////////
	//// Constructor
	// Pre: nh.ok() == true
	// Post: --Calls 'loadCodebook' --Calls 'subscribeToImages'
	FABMapRun::FABMapRun(ros::NodeHandle nh) : OpenFABMap2(nh)
	{

		good_matches = 0;
		counter = self_match_window_;
		num_images = 0;
		last_index = 0;
		stick = 0;
		loop_closures = 0;

		// Load trained data
		bool goodLoad = loadCodebook();
		
		if (goodLoad)
		{
			ROS_INFO("--Codebook successfully loaded!--");
		// Read private parameters
		ros::NodeHandle local_nh_("~");
		local_nh_.param<int>("maxMatches", maxMatches_, 0);
		local_nh_.param<double>("minMatchValue", minMatchValue_, 0.0);
		local_nh_.param<bool>("DisableSelfMatch", disable_self_match_, false);
			local_nh_.param<int>("SelfMatchWindow", self_match_window_, 1);
		local_nh_.param<bool>("DisableUnknownMatch", disable_unknown_match_, false);
			local_nh_.param<bool>("AddOnlyNewPlaces", only_new_places_, false);
		
		// Setup publisher
		pub_ = nh_.advertise<openfabmap2::Match>("appearance_matches",1000);
		
		// Initialise for the first to contain
		// - Match to current
		// - Match to nothing
		confusionMat = cv::Mat::zeros(2,2,CV_64FC1);
		
		// Set callback
//		subscribeToImages();

//		checkXiSquareMatching();

//		checkDescriptors();

//		computeHomography();

		PrecisionRecall();
}
		else
		{
			shutdown();
		}
	}
	
	//// Destructor
	FABMapRun::~FABMapRun()
	{
	}

	void FABMapRun::PrecisionRecall(){
		PrecisionRecall(ORB);
		PrecisionRecall(BRAND);
		PrecisionRecall(SURF);
		PrecisionRecall(SIFT);
		PrecisionRecall(TEST);
	}

	void FABMapRun::PrecisionRecall(eDescriptorType descriptorType){
		std::vector<Stat> mainStats;
		runPRForDescriptorType(descriptorType, mainStats, "5.png", "5_depth.png",
				"200.png", "200_depth.png");
		runPRForDescriptorType(descriptorType, mainStats, "5.png", "5_depth.png",
				"481.png", "481_depth.png");
		runPRForDescriptorType(descriptorType, mainStats, "263.png", "263_depth.png",
				"540.png", "540_depth.png");
		runPRForDescriptorType(descriptorType, mainStats, "222.png", "222_depth.png",
				"516.png", "516_depth.png");
		outputStats(mainStats, descriptorType);
	}

	void FABMapRun::addToMainStats(std::vector<Stat>& stats, std::vector<Stat>& mainStats) {
		bool firstTime = (mainStats.size() == 0);
		for (int i=0; i<stats.size(); i++) {
			if (firstTime) {
				mainStats.push_back(stats[i]);
			}
			else
			{
				mainStats[i].false_neg += stats[i].false_neg;
				mainStats[i].false_pos += stats[i].false_pos;
				mainStats[i].no_index += stats[i].no_index;
				mainStats[i].pos += stats[i].pos;
				mainStats[i].true_pos += stats[i].true_pos;
			}
		}
	}

	void FABMapRun::runPRForDescriptorType(eDescriptorType descriptorType, std::vector<Stat>& mainStats,
			std::string rgb1, std::string depth1, std::string rgb2, std::string depth2){

		std::vector<Stat> stats;
		cv::Mat desc1, desc2;
		std::vector<int> true_index;
		std::vector<cv::KeyPoint> kpts1, kpts2;
		cv::Mat gray, warp_gray;
		cameraFrame frame, warp_frame;

		getKeypoints(kpts1, kpts2, frame, warp_frame, gray, warp_gray, rgb1, depth1, rgb2, depth2);
		getDescriptors(kpts1, kpts2, desc1, desc2, frame, warp_frame, gray, warp_gray, descriptorType);
		getTrueIndex(kpts1, kpts2, desc1, desc2, true_index, descriptorType);
		computePrecisionRecall(desc1, desc2, true_index, descriptorType, stats);
		addToMainStats(stats, mainStats);
	}

	void FABMapRun::getKeypoints(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2,
			 cameraFrame& frame, cameraFrame& warp_frame, cv::Mat& gray, cv::Mat& warp_gray,
			 std::string rgb1, std::string depth1, std::string rgb2, std::string depth2) {

		std::string base = "/home/rasha/Desktop/fabmap/nao_matches/rgbd/";
		std::stringstream ss;
//		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/5.png";
		ss<<base<<rgb1;
		cv::Mat color = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		cv::cvtColor(color, color, CV_BGR2RGB);
		cv::cvtColor(color, gray, CV_RGB2GRAY);
		ss.str("");
//		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/5_depth.png";
		ss<<base<<depth1;
		cv::Mat depth = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.0/25.5);

		ss.str("");
//		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/200.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.430058.png";
		ss<<base<<rgb2;
		cv::Mat warp_color = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		cv::cvtColor(warp_color, warp_color, CV_BGR2RGB);
		cv::cvtColor(warp_color, warp_gray, CV_RGB2GRAY);
		ss.str("");
		ss<<base<<depth2;
//		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/200_depth.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.430172.png";
//		CV_LOAD_IMAGE_GRAYSCALE
		cv::Mat warp_depth = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		warp_depth.convertTo(warp_depth, CV_32F, 1.0/25.5);

		detector2->detect(gray, kpts1);
		detector->detect(warp_gray, kpts2);

		std::cout<<"number of keypoints "<<kpts1.size()<<" "<<kpts2.size()<<std::endl;

		frame = cameraFrame(depth, color);
		warp_frame = cameraFrame(warp_depth, warp_color);
	}

	void FABMapRun::getDescriptors(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2,
			cv::Mat& desc1, cv::Mat& desc2, cameraFrame& frame, cameraFrame& warp_frame,
			cv::Mat& gray, cv::Mat& warp_gray, eDescriptorType descriptorType) {

		if (descriptorType == ORB)
		{
			extractor = new cv::OrbDescriptorExtractor();
			matcher = new cv::BFMatcher(cv::NORM_HAMMING);
		}
		else if (descriptorType == TEST)
		{
			extractor = new NewDesc();
			matcher = new cv::BFMatcher(cv::NORM_HAMMING);
			static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == BRAND)
		{
			extractor = new brand_wrapper();
			matcher = new cv::BFMatcher(cv::NORM_HAMMING);
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == SIFT) {
			extractor = new cv::SIFT();
			matcher = new cv::BFMatcher(cv::NORM_L1);
		}
		else
		{
			extractor = new cv::SURF();
			matcher = new cv::BFMatcher(cv::NORM_L1);
		}

		{
			ScopedTimer timer(__FUNCTION__);
			extractor->compute(gray, kpts1, desc1);
		}

		if (descriptorType == TEST)
		{
			static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = warp_frame;
		}
		else if (descriptorType == BRAND)
		{
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = warp_frame;
		}

		{
			ScopedTimer timer(__FUNCTION__);
			extractor->compute(warp_gray, kpts2, desc2);
		}
	}

	void FABMapRun::getTrueIndex(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2,
			cv::Mat& desc1, cv::Mat& desc2, std::vector<int>& true_index, eDescriptorType descriptorType) {

		true_index = std::vector<int>(desc1.rows, -1);

		std::vector<cv::DMatch> matches;
		std::vector<std::vector<cv::DMatch> > v_matches;
		matcher->knnMatch(desc1, desc2, v_matches, 2);
		for (unsigned int i=0; i<v_matches.size(); i++)
		{
			if (v_matches[i].size() > 1) {
				double dist_perc = (v_matches[i][1].distance - v_matches[i][0].distance)*1.0/v_matches[i][0].distance;
				if (dist_perc >= 0.2) {
					matches.push_back(v_matches[i][0]);
				}
			}
			else {
				std::cout<<"v_matches[i].size() is less than 1 !!!"<<std::endl;
			}
		}

		std::cout<<"matches.size() "<<matches.size()<<std::endl;


		std::vector<cv::Point2f> srcPoints, dstPoints, srcPoints2, dstPoints2;
        for (int i=0; i<matches.size(); i++)
        {
        	srcPoints.push_back(kpts1[matches[i].queryIdx].pt);
        	dstPoints.push_back(kpts2[matches[i].trainIdx].pt);
        }

        cv::Mat homography_mask, homography;
		homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 5, homography_mask);
		std::cout<<homography<<std::endl;

        for (int i=0; i<kpts1.size(); i++)
        {
        	srcPoints2.push_back(kpts1[i].pt);
        }
		perspectiveTransform( srcPoints2, dstPoints2, homography);


		for (int i=0; i<dstPoints2.size(); i++) {
			for (int j=0; j<kpts2.size(); j++) {
				cv::Point2f pt = kpts2[j].pt;
				if (cvRound(pt.x) == cvRound(dstPoints2[i].x) && cvRound(pt.y) == cvRound(dstPoints2[i].y)) {
					true_index[i] = j;
					break;
				}
			}
		}


	}

	void FABMapRun::computePrecisionRecall(cv::Mat& desc1, cv::Mat& desc2, std::vector<int>& true_index,
			eDescriptorType descriptorType, std::vector<Stat>& stats){

/*
		std::ofstream fout_recall("/home/rasha/Desktop/fabmap/plots/recall.txt");
		std::ofstream fout_comp_precision("/home/rasha/Desktop/fabmap/plots/comp_precision.txt");
*/
		int normType;
		std::string descType = "";
		double th_step;
		double th_max;

		if (descriptorType == ORB)
		{
			normType = cv::NORM_HAMMING;
			descType = "ORB";
			th_step = 5;
			th_max = 260;
		}
		else if (descriptorType == TEST)
		{
			normType = cv::NORM_HAMMING;
			descType = "New_desc";
			th_step = 5;
			th_max = 260;
		}
		else if (descriptorType == BRAND)
		{
			normType = cv::NORM_HAMMING;
			descType = "BRAND";
			th_step = 5;
			th_max = 260;
		}
		else if (descriptorType == SIFT) {
			normType = cv::NORM_L1;
			descType = "SIFT";
			th_step = 100;
			th_max = 10000;
		}
		else
		{
			normType = cv::NORM_L1;
			descType = "SURF";
			th_step = 0.1;
			th_max = 10;
		}

		double minDist = std::numeric_limits<double>::max();
		double maxDist = std::numeric_limits<double>::min();

		for (double threshold=0; threshold<th_max; threshold+=th_step) {
			Stat stat;
			stat.pos = 0;
			stat.true_pos = 0;
			stat.false_neg = 0;
			stat.false_pos = 0;
			stat.no_index = 0;
			stat.threshold = threshold;
			for (int i=0; i<desc1.rows; i++) {
				if (true_index[i] == -1)
					stat.no_index++;
				for (int j=0; j<desc2.rows; j++) {
					double dist  = norm(desc1.row(i),desc2.row(j),normType);
					if (dist < minDist)
						minDist = dist;
					if (dist >= maxDist)
						maxDist = dist;
					if (dist <= threshold) {
						stat.pos++;
						if (true_index[i] == j){
							stat.true_pos++;
						}
					} else {
						if (true_index[i] == j){
							stat.false_neg++;
						}
					}
				}
			}

			stat.false_pos = stat.pos - stat.true_pos;
			stats.push_back(stat);

		}

		std::cout<<"minDist "<<minDist<<" maxDist "<<maxDist<<std::endl;

	}

	void FABMapRun::outputStats(std::vector<Stat> stats, eDescriptorType descriptorType){

		std::stringstream ss_recall, ss_precision;
		std::string descType;

		ss_recall<<"/home/rasha/Desktop/fabmap/plots/recall_";
		ss_precision<<"/home/rasha/Desktop/fabmap/plots/comp_precision_";

		if (descriptorType == ORB)
		{
			descType = "ORB";
		}
		else if (descriptorType == TEST)
		{
			descType = "New_desc";
		}
		else if (descriptorType == BRAND)
		{
			descType = "BRAND";
		}
		else if (descriptorType == SIFT) {
			descType = "SIFT";
		}
		else
		{
			descType = "SURF";
		}

		ss_recall<<descType<<".txt";
		ss_precision<<descType<<".txt";

		std::ofstream fout_recall(ss_recall.str().c_str());
		std::ofstream fout_comp_precision(ss_precision.str().c_str());

		std::cout<<"----------running precision_recall for "<<descType<<std::endl;

		double precision, recall, comp_precision;

		for (int i=0; i<stats.size(); i++) {
			Stat stat = stats[i];
			precision = 0;
			if ((stat.true_pos+stat.false_pos) > 0)
				precision = stat.true_pos*1.0/(stat.true_pos+stat.false_pos);
			recall = 0;
			if ((stat.true_pos+stat.false_neg) > 0)
				recall = stat.true_pos*1.0/(stat.true_pos+stat.false_neg);
			comp_precision = 0;
			if ((stat.true_pos+stat.false_pos) > 0)
				comp_precision = stat.false_pos*1.0/(stat.true_pos+stat.false_pos);

			fout_recall<<recall<<std::endl;
			fout_comp_precision<<comp_precision<<std::endl;

			std::cout<<"threshold "<<stat.threshold <<": pos "<<stat.pos<<" true_pos "<<stat.true_pos<<
					" false_pos "<<stat.false_pos<<
					" false_neg "<<stat.false_neg<<" no_index "<<stat.no_index<<" --- "<<
					"precision "<<precision<<
					" 1-precision "<<comp_precision<<
					" recall "<<recall<<std::endl;

		}

		fout_recall.close();
		fout_comp_precision.close();

	}

	void FABMapRun::computePrecisionRecall_old(){
		std::stringstream ss;
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/5.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.398168.png";
		cv::Mat color = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		cv::cvtColor(color, color, CV_BGR2RGB);
		cv::Mat gray;
		cv::cvtColor(color, gray, CV_RGB2GRAY);
		ss.str("");
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/5_depth.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.398210.png";

		cv::Mat depth = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.0/25.5);

		ss.str("");
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/200.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.430058.png";
		cv::Mat warp_color = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		cv::cvtColor(warp_color, warp_color, CV_BGR2RGB);
		cv::Mat warp_gray;
		cv::cvtColor(warp_color, warp_gray, CV_RGB2GRAY);
		ss.str("");
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/200_depth.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.430172.png";
//		CV_LOAD_IMAGE_GRAYSCALE
		cv::Mat warp_depth = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		warp_depth.convertTo(warp_depth, CV_32F, 1.0/25.5);

		std::vector<cv::KeyPoint> kpts, kpts2;
		detector2->detect(gray, kpts);
		detector->detect(warp_gray, kpts2);

		std::cout<<"number of keypoints "<<kpts.size()<<" "<<kpts2.size()<<std::endl;

        cv::Mat desc1, desc2;
		if (descriptorType == BRAND)
		{
			cameraFrame frame(depth, color);
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == CDORB_) {
			cameraFrame frame(depth, color);
			static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == TEST) {
			cameraFrame frame(depth, color);
			static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = frame;
		}

		{
			ScopedTimer timer(__FUNCTION__);
			extractor->compute(gray, kpts, desc1);
		}
		if (descriptorType == BRAND) {
			cameraFrame frame(warp_depth, warp_color);
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == CDORB_) {
			cameraFrame frame(warp_depth, warp_color);
			static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == TEST) {
			cameraFrame frame(warp_depth, warp_color);
			static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = frame;
		}
		{
			ScopedTimer timer(__FUNCTION__);
			extractor->compute(warp_gray, kpts2, desc2);
		}
		std::vector<cv::DMatch> matches, temp_matches1, temp_matches2;

		std::vector<std::vector<cv::DMatch> > v_matches;
		matcher->knnMatch(desc2, desc1, v_matches, 2);
		for (unsigned int i=0; i<v_matches.size(); i++)
		{
			if (v_matches[i].size() > 1) {
				double dist_perc = (v_matches[i][1].distance - v_matches[i][0].distance)*1.0/v_matches[i][0].distance;
//				std::cout<<dist_perc<<std::endl;
				if (dist_perc >= 0.2) {
					matches.push_back(v_matches[i][0]);
					std::cout<<"v_matches "<<i<<" "<<matches[matches.size()-1].trainIdx<<
							" "<<matches[matches.size()-1].queryIdx<<std::endl;
				}
			}
		}
/*
		matcher->match(desc2, desc1, temp_matches1);
		matcher->match(desc1, desc2, temp_matches2);

		for (int i=0; i<temp_matches1.size(); i++) {
			int other_index = temp_matches1[i].trainIdx;
			if (temp_matches2[other_index].trainIdx == temp_matches1[i].queryIdx)
				matches.push_back(temp_matches1[i]);
		}
*/
		std::vector<cv::Point2f> srcPoints, dstPoints;
        for (int i=0; i<matches.size(); i++)
        {
        	srcPoints.push_back(kpts[matches[i].trainIdx].pt);
        	dstPoints.push_back(kpts2[matches[i].queryIdx].pt);
        	std::cout<<i<<" "<<matches[i].distance<<std::endl;
        }

        cv::Mat homography_mask, homography;
		homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 5, homography_mask);

		std::cout<<homography<<std::endl;

		std::vector<cv::Point2f> dstPoints2(srcPoints.size());
		perspectiveTransform( srcPoints, dstPoints2, homography);

		std::vector<cv::KeyPoint> kpts_h;
		std::vector<int> index;

		for (int j=0; j<dstPoints2.size(); j++) {
			std::cout<<dstPoints2[j].x<<" "<<dstPoints2[j].y<<" "<<
					dstPoints[j].x<<" "<<dstPoints[j].y<<" "<<srcPoints[j].x<<" "<<srcPoints[j].y<<std::endl;
		for (int i=0; i<kpts2.size(); i++) {
			cv::Point2f pt = kpts2[i].pt;
				if (cvRound(pt.x) == cvRound(dstPoints2[j].x) && cvRound(pt.y) == cvRound(dstPoints2[j].y)) {
					kpts_h.push_back(kpts2[i]);
					index.push_back(j);
					break;
				}
			}
		}
		cv::Mat desc_h;
		extractor->compute(warp_gray, kpts_h, desc_h);

		std::cout<<"dstPoints2.size() "<<dstPoints2.size()<<" kpts_h.size() "<<kpts_h.size()<<std::endl;

		double threshold = 40;
		int pos = 0;
		int tru_pos = 0;
		int false_neg = 0;
		for (int i=0; i<1; i++) { //for (int i=0; i<desc1.rows; i++) {
			std::cout<<"i "<<i<<" v_matches[i].trainIdx "<<v_matches[i][0].trainIdx<<
					" v_matches[i].queryIdx "<<v_matches[i][0].queryIdx<<" v_matches[i].distance "<<v_matches[i][0].distance<<std::endl;
			for (int j=0; j<desc_h.rows; j++) { //for (int j=0; j<desc_h.rows; j++) {
				std::cout<<"index[j] "<<index[j]<<" matches[index[j]].trainIdx "<<matches[index[j]].trainIdx<<
						" matches[index[j]].queryIdx "<<matches[index[j]].queryIdx<<" matches[index[j]].distance "<<
						matches[index[j]].distance<<std::endl;
				double dist  = norm(desc1.row(i),desc_h.row(j),cv::NORM_HAMMING);
				std::cout<<"i "<<i<<" j "<<j<<" dist "<<dist<<std::endl;
				if (dist <= threshold) {
					pos++;
					if (index[j] == i){
						tru_pos++;
						std::cout<<i<<" true "<<dist<<std::endl;
					}
				} else {
					if (index[j] == i){
						false_neg++;
						std::cout<<i<<" false "<<dist<<std::endl;
					}
				}
			}
		}

		std::cout<<"pos "<<pos<<" tru_pos "<<tru_pos<<" false_neg "<<false_neg<<std::endl;


/*
		cv::Mat matches_img;
		drawMatches(warp_depth, kpts2, depth, kpts, matches, matches_img, cv::Scalar::all(-1), cv::Scalar::all(-1), matches_mask);
		cv::imshow("matches", matches_img);
		cv::waitKey(0);
*/
/*
		int sum_h = 0;
		for (int i=0; i<matches.size(); i++)
			sum_h += (int)homography_mask.at<uchar>(i);

		double inliers = sum(homography_mask)[0];
		std::cout<<"inliers/matches "<<inliers<<"/"<<matches.size()<<" "<<inliers/matches.size()<<std::endl;
		std::cout<<sum_h<<std::endl;
*/
/*
		cv::Mat matches_img;
		drawMatches(warp_depth, kpts2, depth, kpts, matches, matches_img);
		cv::imshow("matches", matches_img);
		cv::waitKey(0);
*/
	}

	void FABMapRun::computeHomography(){
		std::stringstream ss;
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/222.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.398168.png";
		cv::Mat color = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		cv::cvtColor(color, color, CV_BGR2RGB);
		cv::Mat gray;
		cv::cvtColor(color, gray, CV_RGB2GRAY);
		ss.str("");
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/222_depth.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.398210.png";

		cv::Mat depth = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		depth.convertTo(depth, CV_32F, 1.0/25.5);

		ss.str("");
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/516.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.430058.png";
		cv::Mat warp_color = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		cv::cvtColor(warp_color, warp_color, CV_BGR2RGB);
		cv::Mat warp_gray;
		cv::cvtColor(warp_color, warp_gray, CV_RGB2GRAY);
		ss.str("");
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/516_depth.png";
//		ss<<"/home/rasha/Desktop/fabmap/pioneer_360/1311876800.430172.png";
//		CV_LOAD_IMAGE_GRAYSCALE
		cv::Mat warp_depth = cv::imread(ss.str(), CV_LOAD_IMAGE_UNCHANGED);
		warp_depth.convertTo(warp_depth, CV_32F, 1.0/25.5);

		std::vector<cv::KeyPoint> kpts, kpts2;
		detector2->detect(gray, kpts);
		detector->detect(warp_gray, kpts2);

		std::cout<<"number of keypoints "<<kpts.size()<<" "<<kpts2.size()<<std::endl;

        cv::Mat desc1, desc2;
		if (descriptorType == BRAND)
		{
			cameraFrame frame(depth, color);
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == CDORB_) {
			cameraFrame frame(depth, color);
			static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == TEST) {
			cameraFrame frame(depth, color);
			static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = frame;
		}

		{
			ScopedTimer timer(__FUNCTION__);
			extractor->compute(gray, kpts, desc1);
		}
		if (descriptorType == BRAND) {
			cameraFrame frame(warp_depth, warp_color);
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == CDORB_) {
			cameraFrame frame(warp_depth, warp_color);
			static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == TEST) {
			cameraFrame frame(warp_depth, warp_color);
			static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = frame;
		}
		{
			ScopedTimer timer(__FUNCTION__);
			extractor->compute(warp_gray, kpts2, desc2);
		}
		std::vector<cv::DMatch> matches, temp_matches1, temp_matches2;

		std::vector<std::vector<cv::DMatch> > v_matches;
		matcher->knnMatch(desc2, desc1, v_matches, 2);
		for (unsigned int i=0; i<v_matches.size(); i++)
		{
			if (v_matches[i].size() > 1) {
				double dist_perc = (v_matches[i][1].distance - v_matches[i][0].distance)*1.0/v_matches[i][0].distance;
				std::cout<<dist_perc<<std::endl;
				if (dist_perc >= 0.2)
					matches.push_back(v_matches[i][0]);
			}
		}
/*
		matcher->match(desc2, desc1, temp_matches1);
		matcher->match(desc1, desc2, temp_matches2);

		for (int i=0; i<temp_matches1.size(); i++) {
			int other_index = temp_matches1[i].trainIdx;
			if (temp_matches2[other_index].trainIdx == temp_matches1[i].queryIdx)
				matches.push_back(temp_matches1[i]);
		}
*/
		std::vector<cv::Point2f> srcPoints, dstPoints;
        for (int i=0; i<matches.size(); i++)
        {
        	srcPoints.push_back(kpts[matches[i].trainIdx].pt);
        	dstPoints.push_back(kpts2[matches[i].queryIdx].pt);
        }

        cv::Mat homography_mask, homography;
		homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 5, homography_mask);

		int sum_h = 0;
		for (int i=0; i<matches.size(); i++)
			sum_h += (int)homography_mask.at<uchar>(i);

		double inliers = sum(homography_mask)[0];
		std::cout<<"inliers/matches "<<inliers<<"/"<<matches.size()<<" "<<inliers/matches.size()<<std::endl;
		std::cout<<sum_h<<std::endl;
/*
		cv::Mat matches_img;
		drawMatches(warp_depth, kpts2, depth, kpts, matches, matches_img);
		cv::imshow("matches", matches_img);
		cv::waitKey(0);
*/
	}

	void FABMapRun::checkDescriptors(){
		std::stringstream ss;
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/5.png";
		cv::Mat color = cv::imread(ss.str());
		cv::cvtColor(color, color, CV_BGR2RGB);
		cv::Mat gray;
		cv::cvtColor(color, gray, CV_RGB2GRAY);
//		ss.str("");
//		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd_gray/5.png";
//		cv::Mat gray = cv::imread(ss.str(), CV_LOAD_IMAGE_GRAYSCALE);
		ss.str("");
		ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/5_depth.png";
		cv::Mat depth = cv::imread(ss.str(), CV_LOAD_IMAGE_GRAYSCALE);

		cv::Mat  warp_gray = cv::Mat::zeros( gray.rows, gray.cols, gray.type() );
		cv::Mat  warp_depth = cv::Mat::zeros( depth.rows, depth.cols, depth.type() );
		cv::Mat  warp_color = cv::Mat::zeros( color.rows, color.cols, color.type() );

		cv::Point center = cv::Point( gray.cols/2, gray.rows/2 );
		double angle = 30; //30.0;
		double scale = 1.0;
//		cv::Mat rot_mat = getRotationMatrix2D( center, angle, scale );

		double angle_rad = angle * CV_PI/180;
        cv::Mat rot_mat = (cv::Mat_<float>(2, 3) <<   cos(angle_rad), -sin(angle_rad), 0,
                                                sin(angle_rad),  cos(angle_rad), 0);

//		std::cout<<"rotation matrix "<<std::endl<<rot_mat<<std::endl;

		warpAffine(gray, warp_gray, rot_mat, warp_gray.size() );
		warpAffine(depth, warp_depth, rot_mat, warp_depth.size() );
		warpAffine(color, warp_color, rot_mat, warp_color.size() );
		std::vector<cv::KeyPoint> kpts, kpts2;
		detector2->detect(gray, kpts);
		detector->detect(warp_gray, kpts2);

/*
		cv::Mat depth_kp;
		drawKeypoints(warp_depth, kpts2, depth_kp);
		cv::imshow("depth_kp", depth_kp);
		cv::waitKey(0);
*/

        cv::Mat desc1, desc2;
		if (descriptorType == BRAND)
		{
			cameraFrame frame(depth, color);
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == CDORB_) {
			cameraFrame frame(depth, color);
			static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
		}

		extractor->compute(gray, kpts, desc1);
		if (descriptorType == BRAND) {
			cameraFrame frame(warp_depth, warp_color);
			static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
		}
		else if (descriptorType == CDORB_) {
			cameraFrame frame(warp_depth, warp_color);
			static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
		}
		extractor->compute(warp_gray, kpts2, desc2);

		std::vector<cv::DMatch> matches;
		matcher->match(desc2, desc1, matches);

//		std::cout<<kpts.size()<<" "<<kpts2.size()<<" "<<matches.size()<<std::endl;
		std::vector<char>matches_mask(matches.size(), 0);

        double inliers = 0;
        double avg_dist = 0;
        for (int i=0; i<matches.size(); i++)
        {
        	cv::Point p1 = kpts[matches[i].trainIdx].pt;
        	std::cout<<i<<": "<<p1;
 //       	circle(gray, p1, 2, cv::Scalar(255,0,0));
        	float temp = p1.x;
        	p1.x = p1.x*rot_mat.at<float>(0,0) + p1.y*rot_mat.at<float>(0,1) + rot_mat.at<float>(0,2);
        	p1.y = temp*rot_mat.at<float>(1,0) + p1.y*rot_mat.at<float>(1,1) + rot_mat.at<float>(1,2);
        	cv::Point p2 = kpts2[matches[i].queryIdx].pt;
/*        	circle(warp_gray, p1, 4, cv::Scalar(255,0,0));
        	circle(warp_gray, p2, 2, cv::Scalar(0,255,0));
    		cv::imshow("gray", gray);
        	cv::imshow("warp", warp_gray);
    		cv::waitKey(0);
*/        	double error = cv::norm(p1-p2);
        	std::cout<<" rot "<<p1<<" det2 "<<p2<<" error "<<error<<" dist "<<matches[i].distance<<std::endl;
        	if(error < 5) {
        		inliers+=1;
        		avg_dist += matches[i].distance;
        	}
        	else {
        		matches_mask[i] = 1;
        	}
        }
        std::cout<<"inliers/matches "<<inliers<<"/"<<matches.size()<<" "<<inliers/matches.size()<<
        		" avg_dist "<<avg_dist/inliers<<std::endl;

		cv::Mat matches_img;
		drawMatches(warp_depth, kpts2, depth, kpts, matches, matches_img, cv::Scalar::all(-1), cv::Scalar::all(-1), matches_mask);
		cv::imshow("matches", matches_img);
		cv::waitKey(0);


	}

	void FABMapRun::checkXiSquareMatching() {
		CV_Assert(trainbows.type() == CV_32F);
		double sum1 = 0;
		double sum2 = 0;
		for (int i=1; i<trainbows.rows; i++)
		{
			double min1 = std::numeric_limits<double>::max();
			double min2 = std::numeric_limits<double>::max();
			int index1 = 0;
			int index2 = 0;
			for (int j=0; j<i; j++)
			{
				double cmp1 = compareHist(trainbows.row(i), trainbows.row(j), CV_COMP_CHISQR);
				double cmp2 = compareHist(trainbows.row(i), trainbows.row(j), CV_COMP_BHATTACHARYYA);
//				std::cout<<cmp1<<" ";
				if (cmp1 < min1) {
					min1 = cmp1;
					index1 = j;
				}
				if (cmp2 < min2) {
					min2 = cmp2;
					index2 = j;
				}

			}
			std::cout<<"i "<<i<<" min1 "<<min1<<" j "<<index1<<" min2 "<<min2
					<<" j "<<index2<<std::endl;
			sum1 += min1;
			sum2 += min2;
		}
		std::cout<<"sum1, avg1 "<<sum1<<" "<<sum1/(trainbows.rows-1)<<" sum2, avg2 "<<sum2<<" "<<
				sum2/(trainbows.rows-1)<<std::endl;
	}

	//// Image Callback
	// Pre: image_msg->encoding == end::MONO8
	// Post: -Matches to 'image_msg' published on pub_
	//			 -'firstFrame_' blocks initial nonsensical self match case
void FABMapRun::processImgCallback(const sensor_msgs::ImageConstPtr& image_msg) {
	ROS_DEBUG_STREAM("OpenFABMap2-> Processing image sequence number: " << image_msg->header.seq);
	cv_bridge::CvImagePtr cv_ptr;
	try {
		// TODO: toCvShare should be used for 'FABMapRun'
		cv_ptr = cv_bridge::toCvCopy(image_msg, enc::MONO8);
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	cameraFrame currentFrame(cv_ptr);
	processImage(currentFrame);

}

void FABMapRun::processImgCallback(const sensor_msgs::ImageConstPtr& image_msg,
		const sensor_msgs::ImageConstPtr& depth_msg) {
	ROS_DEBUG_STREAM("OpenFABMap2-> Processing image sequence number: " << image_msg->header.seq);
	cv_bridge::CvImagePtr cv_ptr;
	cv_bridge::CvImagePtr cv_depth_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(image_msg);
		cv_depth_ptr = cv_bridge::toCvCopy(depth_msg);//, enc::MONO8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}


	sensor_msgs::CameraInfoConstPtr cam_info_msg;


	cameraFrame frame(cv_ptr, cv_depth_ptr, cam_info_msg);
//	cv_depth_ptr->image.convertTo(frame.depth_img, CV_8UC1, 25.5); //100,0); //TODO: change value

/*
	double min, max;
	cv::Point minL, maxL;
	cv::minMaxLoc(cv_depth_ptr->image, &min, &max, &minL, &maxL, frame.depth_img);
	if (min < g_min)
		g_min = min;
	if (max > g_max)
		g_max = max;
	std::cout<<"min "<<min<<" max "<<max<<" g_min "<<g_min<<" g_max "<<g_max<<std::endl;

	return;
*/

/*
	if (descriptorType == BRAND)
		static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
	else if (descriptorType == CDORB_)
		static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
*/
	processImage(frame);
}

void FABMapRun::processImage(cameraFrame& frame) {

/*
	cv::Mat rgb;
	std::stringstream ss;
	ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/"<<(num_images)<<".png";
	cv::cvtColor(frame.image_ptr->image, rgb, CV_BGR2RGB);
	cv::imwrite(ss.str(), rgb);
	ss.str("");
	ss<<"/home/rasha/Desktop/fabmap/nao_matches/rgbd/"<<(num_images)<<"_depth.png";
	cv::imwrite(ss.str(), frame.depth_img);

	std::cout<<num_images<<std::endl;
*/

	num_images++;

//	return;

/*
	if (counter >= self_match_window_)
	{
		counter = 0;
	}
	else
	{
		counter++;
		return;
	}
*/

//	counter = self_match_window_;

	ROS_DEBUG("Received %d by %d image, depth %d, channels %d", frame.color_img.cols, frame.color_img.rows, frame.color_img.depth(), frame.color_img.channels());

	cv::Mat bow;
	ROS_DEBUG("Detector.....");
	detector->detect(frame.color_img, kpts);


	std::cout<<"size of kpts "<<kpts.size()<<std::endl;
	ROS_DEBUG("Compute discriptors...");


	if (descriptorType == BRAND)
		static_cast<cv::Ptr<brand_wrapper> >(extractor)->currentFrame = frame;
	else if (descriptorType == CDORB_)
		static_cast<cv::Ptr<CDORB> >(extractor)->currentFrame = frame;
	else if (descriptorType == TEST)
		static_cast<cv::Ptr<NewDesc> >(extractor)->currentFrame = frame;


	std::vector<std::vector<int> > pointIdxsOfClusters;
	cv::Mat desc;
	bide->compute(frame.color_img, kpts, bow, &pointIdxsOfClusters, &desc);

	int fromImageIndex;

	// Check if the frame could be described
	if (!bow.empty() && kpts.size() > minDescriptorCount_) {
		// IF NOT the first frame processed
		if (!firstFrame_) {
			ROS_DEBUG("Compare bag of words...");
			std::vector<of2::IMatch> matches;

			fromImageIndex = fabMap->getTestImgDescriptorsCount();

			// Find match likelyhoods for this 'bow'
			fabMap->compare(bow, matches, !only_new_places_);

			// Sort matches with oveloaded '<' into
			// Accending 'match' order
			std::sort(matches.begin(), matches.end());

			// Add BOW
			if (!only_new_places_)
				ROS_ERROR("only_new_places is false");
			else
			{
				of2::IMatch bestMatch = matches.back();

				if (bestMatch.imgIdx == last_index)
						stick += 1.0;

				int loc_img;
				if (bestMatch.imgIdx == -1) {
					ROS_WARN_STREAM("Adding bow of new place...");
					fabMap->add(bow);
					location_image[fromImageIndex] = num_images - 1;
/*					std::stringstream ss;
					ss<<"/home/rasha/Desktop/fabmap/nao_matches/new_places/"<<(fromImageIndex)<<".png";
					cv::imwrite(ss.str(), frame.image_ptr->image);
*/					loc_img = -1;
				}
				else
				{
					loc_img = location_image[bestMatch.imgIdx];
					if (bestMatch.match >= 0.98)
							good_matches += 1.0;
					loop_closures += 1.0;
				}


				ROS_INFO_STREAM("image_number "<< num_images-1<<
					       " toLocation " << bestMatch.imgIdx <<
						  " Match "<< bestMatch.match <<
						  " good LC "<< good_matches / ((loop_closures==0)? 1:loop_closures)
//						  " good_matches "<< good_matches / (num_images-1) <<
//						  " stick "<< stick / (num_images-1)
				);

				last_index = bestMatch.imgIdx;
			}

			if (visualise_)
				visualiseMatches2(matches);


		}
		else {
			std::cout<<"kpt"<<std::endl;
			std::cout<<kpts[0].pt.x<<" "<<kpts[0].pt.y<<std::endl;
			std::cout<<"desc"<<std::endl;
			std::cout<<desc.row(0)<<std::endl;
			std::cout<<"bow"<<std::endl;
			std::cout<<bow.row(0)<<std::endl;

			ROS_WARN_STREAM("Adding bow of new place...");
			fabMap->add(bow);
			location_image[0] = 0;
			std::stringstream ss;
			ss<<"/home/rasha/Desktop/fabmap/nao_matches/new_places/0.png";
			cv::imwrite(ss.str(), frame.color_img);
			firstFrame_ = false;
			last_index = -1;
		}
	} else {
		ROS_WARN("--Image not descriptive enough, ignoring.");
	}


}

//// Visualise Matches
// Pre:
// Post:
void FABMapRun::visualiseMatches2(std::vector<of2::IMatch> &matches)
{
	int numImages = num_images;

	if (confusionMat.cols < numImages) {
		cv::Mat newConfu = cv::Mat::zeros(numImages+10,numImages+10, CV_64FC1);
		cv::Mat roi(newConfu, cv::Rect(0,0,confusionMat.cols,confusionMat.rows));

		confusionMat.copyTo(roi);
		confusionMat = newConfu.clone();
	}

	for (std::vector<of2::IMatch>::reverse_iterator matchIter = matches.rbegin();
			matchIter != matches.rend(); ++matchIter) {

		if (matchIter->imgIdx == -1)
			confusionMat.at<double>(numImages-1, numImages-1) = 255*(double)matchIter->match;
		else
			confusionMat.at<double>(numImages-1, location_image[matchIter->imgIdx]) = 255*(double)matchIter->match;

		break;

	}

	cv::imshow("Confusion Matrix", confusionMat);
	cv::waitKey(10);
}
	
	//// Visualise Matches
	// Pre:
	// Post:
	void FABMapRun::visualiseMatches(std::vector<of2::IMatch> &matches)
	{
	//	return;
		int numMatches = matches.size();
		
		cv::Mat newConfu = cv::Mat::zeros(numMatches,numMatches, CV_64FC1);
		ROS_DEBUG_STREAM("'newConfu -> rows: " << newConfu.rows
										<< " cols: " << newConfu.cols);
		cv::Mat roi(newConfu, cv::Rect(0,0,confusionMat.cols,confusionMat.rows));
		ROS_DEBUG_STREAM("'ROI -> rows: " << roi.rows
										<< " cols: " << roi.cols);
		confusionMat.copyTo(roi);
		
		for (std::vector<of2::IMatch>::reverse_iterator matchIter = matches.rbegin();
				 matchIter != matches.rend();
				 ++matchIter) 
		{
			// Skip null match
			if (matchIter->imgIdx == -1)
			{
				continue;
			}
			
			ROS_DEBUG_STREAM("QueryIdx " << matchIter->queryIdx <<
											 " ImgIdx " << matchIter->imgIdx <<
											 " Likelihood " << matchIter->likelihood <<
											 " Match " << matchIter->match);
			
			ROS_DEBUG_STREAM("--About to multi " << 255 << " by " << (double)matchIter->match);
			ROS_DEBUG_STREAM("---Result " << floor(255*((double)matchIter->match)));
			newConfu.at<double>(numMatches-1, matchIter->imgIdx) = 255*(double)matchIter->match;
			ROS_DEBUG_STREAM("-Uchar: " << newConfu.at<double>(numMatches-1, matchIter->imgIdx)
											<< " at (" << numMatches << ", " << matchIter->imgIdx << ")");
		}
		newConfu.at<double>(numMatches-1, numMatches-1) = 255.0;
		ROS_DEBUG_STREAM("-Value: " << newConfu.at<double>(numMatches-1,numMatches-1)
										<< " at (" << numMatches << ", " << numMatches << ")");
		
		confusionMat = newConfu.clone();
		ROS_DEBUG_STREAM("'confusionMat -> rows: " << confusionMat.rows
										<< " cols: " << confusionMat.cols);
		
		cv::imshow("Confusion Matrix", newConfu);
		cv::waitKey(10);
	}
	
	//// File loader
	// Pre:
	// Post:
	bool FABMapRun::loadCodebook()
	{
		ROS_INFO("Loading codebook...");
		
		cv::FileStorage fs;
		
		fs.open(vocabPath_,
						cv::FileStorage::READ);
		fs["Vocabulary"] >> vocab;
		fs.release();
		ROS_INFO("Vocabulary with %d words, %d dims loaded",vocab.rows,vocab.cols);
		
		fs.open(clTreePath_,
						cv::FileStorage::READ);
		fs["Tree"] >> clTree;
		fs.release();
		ROS_INFO("Chow Liu Tree loaded");
		
		fs.open(trainbowsPath_,
						cv::FileStorage::READ);
		fs["Trainbows"] >> trainbows;
		fs.release();
		ROS_INFO("Trainbows loaded");
		
		ROS_INFO("Setting the Vocabulary...");
		bide->setVocabulary(vocab);
		
		ROS_INFO("Initialising FabMap2 with Chow Liu tree...");
		
		// Get additional parameters
		ros::NodeHandle local_nh_("~");
		
		//create options flags
		std::string new_place_method, bayes_method;
		int simple_motion;
		local_nh_.param<std::string>("NewPlaceMethod", new_place_method, "Meanfield");
		local_nh_.param<std::string>("BayesMethod", bayes_method, "ChowLiu");
		local_nh_.param<int>("SimpleMotion", simple_motion, 0);
		
		int options = 0;
		if(new_place_method == "Sampled") {
			options |= of2::FabMap::SAMPLED;
		} else {
			options |= of2::FabMap::MEAN_FIELD;
		}
		if(bayes_method == "ChowLiu") {
			options |= of2::FabMap::CHOW_LIU;
		} else {
			options |= of2::FabMap::NAIVE_BAYES;
		}
		if(simple_motion) {
			options |= of2::FabMap::MOTION_MODEL;
		}
		
		//create an instance of the desired type of FabMap
		std::string fabMapVersion;
		double pzge, pzgne;
		int num_samples;
		local_nh_.param<std::string>("FabMapVersion", fabMapVersion, "FABMAPFBO");
		local_nh_.param<double>("PzGe", pzge, 0.39);
		local_nh_.param<double>("PzGne", pzgne, 0);
		local_nh_.param<int>("NumSamples", num_samples, 3000);
		
		if(fabMapVersion == "FABMAP1") {			
			fabMap = new of2::FabMap1(clTree, 
																pzge,
																pzgne,
																options,
																num_samples);
			
		} else if(fabMapVersion == "FABMAPLUT") {
			int lut_precision;
			local_nh_.param<int>("Precision", lut_precision, 6);
			fabMap = new of2::FabMapLUT(clTree, 
																	pzge,
																	pzgne,
																	options,
																	num_samples,
																	lut_precision);
			
		} else if(fabMapVersion == "FABMAPFBO") {
			double fbo_rejection_threshold, fbo_psgd;
			int fbo_bisection_start, fbo_bisection_its;
			local_nh_.param<double>("RejectionThreshold", fbo_rejection_threshold, 1e-6);
			local_nh_.param<double>("PsGd", fbo_psgd, 1e-6);
			local_nh_.param<int>("BisectionStart", fbo_bisection_start, 512);
			local_nh_.param<int>("BisectionIts", fbo_bisection_its, 9);
			fabMap = new of2::FabMapFBO(clTree, 
																	pzge,
																	pzgne,
																	options,
																	num_samples,
																	fbo_rejection_threshold,
																	fbo_psgd,
																	fbo_bisection_start,
																	fbo_bisection_its);
			
		} else if(fabMapVersion == "FABMAP2") {
			fabMap = new of2::FabMap2(clTree, 
																pzge,
																pzgne,
																options);
		} else {
			ROS_ERROR("Could not identify openFABMAPVersion from node params");
			return false;
		}
		
		ROS_INFO("Adding the trained bag of words...");
		fabMap->addTraining(trainbows);
		
		return true;
	}
	
	//// Unlink Callback
	// Pre:
	// Post: --Cleanup
	void FABMapRun::shutdown()
	{
		// Flag this worker as complete
		working_ = false;
		
		if (sub_.getNumPublishers() > 0)
		{
			ROS_WARN_STREAM("Shutting down " << sub_.getNumPublishers() << " subscriptions...");
			sub_.shutdown();
			nh_.shutdown();
		}
		else
		{
			ROS_ERROR_STREAM("-> " << sub_.getNumPublishers() << " subscriptions when shutting down..");
		}
	}
	// end class implementation FABMapRun
} // end namespace openfabmap2_ros
