/*
 * BOWMSCBinaryTrainer.cpp
 *
 *  Created on: Aug 5, 2016
 *      Author: rasha
 */

#include "openfabmap.hpp"

using std::vector;
using std::list;
using cv::Mat;

namespace of2 {

BOWMSCBinaryTrainer::BOWMSCBinaryTrainer(double _clusterSize) :
	clusterSize(_clusterSize) {
}

BOWMSCBinaryTrainer::~BOWMSCBinaryTrainer() {
}

Mat BOWMSCBinaryTrainer::cluster() const {
	CV_Assert(!descriptors.empty());
	int descCount = 0;
	for(size_t i = 0; i < descriptors.size(); i++)
	descCount += descriptors[i].rows;

	Mat mergedDescriptors(descCount, descriptors[0].cols,
		descriptors[0].type());
	for(size_t i = 0, start = 0; i < descriptors.size(); i++)
	{
		Mat submut = mergedDescriptors.rowRange((int)start,
			(int)(start + descriptors[i].rows));
		descriptors[i].copyTo(submut);
		start += descriptors[i].rows;
	}
	return cluster(mergedDescriptors);
}

static inline int hamming_distance_orb32x8_popcountll(const uint64_t* v1, const uint64_t* v2) {
  return (__builtin_popcountll(v1[0] ^ v2[0]) + __builtin_popcountll(v1[1] ^ v2[1])) +
         (__builtin_popcountll(v1[2] ^ v2[2]) + __builtin_popcountll(v1[3] ^ v2[3]));
}

int bruteForceSearchORB(const uint64_t* v, const std::vector<cv::Mat>& initialCentres, const unsigned int& size, int& result_index){
  //constexpr unsigned int howmany64bitwords = 4;//32*8/64;
	const unsigned int howmany64bitwords = 4;//32*8/64;
//  assert(search_array && "Nullpointer in bruteForceSearchORB");
  result_index = -1;//impossible
  int min_distance = 1 + 256;//More than maximum distance
  for(unsigned int i = 0; i < size; i+=1){
	uint64_t* search_array = reinterpret_cast<uint64_t*>(initialCentres[i].data);
    int hamming_distance_i = hamming_distance_orb32x8_popcountll(v, search_array);
//    std::cout<<"c "<<i<<" dist "<<hamming_distance_i<<std::endl;
    if(hamming_distance_i < min_distance){
      min_distance = hamming_distance_i;
      result_index = i;
    }
  }
  return min_distance;
}

Mat BOWMSCBinaryTrainer::cluster(const Mat& descriptors) const {

	CV_Assert(!descriptors.empty());

	// TODO: sort the descriptors before clustering.

	vector<Mat> initialCentres;

//	initialCentres.push_back(descriptors.row(0));
	Mat centre1 = Mat::zeros(1,descriptors.cols,descriptors.type());
	Mat centre2 = Mat::zeros(1,descriptors.cols,descriptors.type());
	centre2.setTo(255);
//	std::cout<<"centre1 "<<centre1<<std::endl;
//	std::cout<<"centre2 "<<centre2<<std::endl;
	initialCentres.push_back(centre1);
	initialCentres.push_back(centre2);
	uint64_t* query_value =  reinterpret_cast<uint64_t*>(descriptors.data);
//	uint64_t* search_array = reinterpret_cast<uint64_t*>(initialCentres.data());
	for(unsigned int i = 0; i < descriptors.rows; ++i, query_value += 4){//ORB feature = 32*8bit = 4*64bit
	  int result_index = -1;
	  int hd = bruteForceSearchORB(query_value, initialCentres, initialCentres.size(), result_index);
		if (hd > clusterSize) {
			initialCentres.push_back(descriptors.row(i));
//			search_array = reinterpret_cast<uint64_t*>(initialCentres.data());
		}
	}

	vector<vector<Mat> > clusters;
	clusters.resize(initialCentres.size());
	query_value =  reinterpret_cast<uint64_t*>(descriptors.data);
//	search_array = reinterpret_cast<uint64_t*>(initialCentres.data());
	for(unsigned int i = 0; i < descriptors.rows; ++i, query_value += 4){//ORB feature = 32*8bit = 4*64bit
	  int result_index = -1;
	  int hd = bruteForceSearchORB(query_value, initialCentres, initialCentres.size(), result_index);
	  clusters[result_index].push_back(descriptors.row(i));
	}

	// TODO: throw away small clusters.


	//Author: Dorian Galvez-Lopez

	Mat vocabulary;
	for (size_t c = 0; c < clusters.size(); c++) {
		vector<int> sum(descriptors.cols * 8, 0);
		for(size_t i = 0; i < clusters[c].size(); ++i)
		{
		  const cv::Mat &d = clusters[c][i];
		  const unsigned char *p = d.ptr<unsigned char>();

		  for(int j = 0; j < d.cols; ++j, ++p)
		  {
			if(*p & (1 << 7)) ++sum[ j*8     ];
			if(*p & (1 << 6)) ++sum[ j*8 + 1 ];
			if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
			if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
			if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
			if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
			if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
			if(*p & (1))      ++sum[ j*8 + 7 ];
		  }
		}

		Mat centre = cv::Mat::zeros(1, descriptors.cols, CV_8U);
		unsigned char *p = centre.ptr<unsigned char>();

		const int N2 = (int)clusters[c].size() / 2 + clusters[c].size() % 2;
		for(size_t i = 0; i < sum.size(); ++i)
		{
		  if(sum[i] >= N2)
		  {
			// set bit
			*p |= 1 << (7 - (i % 8));
		  }

		  if(i % 8 == 7) ++p;
		}
		vocabulary.push_back(centre);
	}

	return vocabulary;

}

}




