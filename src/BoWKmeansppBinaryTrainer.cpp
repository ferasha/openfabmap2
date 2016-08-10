/*
 * BoWKmeansppBinaryTrainer.cpp
 *
 *  Created on: Aug 10, 2016
 *      Author: rasha
 */

#include "openfabmap.hpp"
//#include "precomp.hpp"

namespace of2 {

using namespace cv;

static inline int hamming_distance_orb32x8_popcountll(const uint64_t* v1, const uint64_t* v2) {
  return (__builtin_popcountll(v1[0] ^ v2[0]) + __builtin_popcountll(v1[1] ^ v2[1])) +
         (__builtin_popcountll(v1[2] ^ v2[2]) + __builtin_popcountll(v1[3] ^ v2[3]));
}

float hammingDist(const uchar* sample, const uchar* center, const int dims){
	CV_Assert(dims == 32);
	const uint64_t* sample_ =  reinterpret_cast<const uint64_t*>(sample);
	const uint64_t* center_ =  reinterpret_cast<const uint64_t*>(center);
	return hamming_distance_orb32x8_popcountll(sample_, center_);
}

class KMeansBinaryPPDistanceComputer : public ParallelLoopBody
{
public:
    KMeansBinaryPPDistanceComputer( float *_tdist2,
                              const uchar *_data,
                              const float *_dist,
                              int _dims,
                              size_t _step,
                              size_t _stepci )
        : tdist2(_tdist2),
          data(_data),
          dist(_dist),
          dims(_dims),
          step(_step),
          stepci(_stepci) { }

    void operator()( const cv::Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;

        for ( int i = begin; i<end; i++ )
        {
            tdist2[i] = std::min(hammingDist(data + step*i, data + stepci, dims), dist[i]);
        }
    }

private:
    KMeansBinaryPPDistanceComputer& operator=(const KMeansBinaryPPDistanceComputer&); // to quiet MSVC

    float *tdist2;
    const uchar *data;
    const float *dist;
    const int dims;
    const size_t step;
    const size_t stepci;
};

/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
static void generateCentersPP(const Mat& _data, Mat& _out_centers,
                              int K, RNG& rng, int trials)
{
    int i, j, k, dims = _data.cols, N = _data.rows;
    const uchar* data = _data.ptr<uchar>(0);
    size_t step = _data.step/sizeof(data[0]);
    std::vector<int> _centers(K-2);
    int* centers = &_centers[0];
    std::vector<float> _dist(N*3);
    float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
    double sum0 = 0;

 //   centers[0] = (unsigned)rng % N;

	Mat centre1 = Mat::zeros(1,dims,CV_8U);
	Mat centre2 = Mat::zeros(1,dims,CV_8U);
	centre2.setTo(255);

	uchar* centre1_ptr = centre1.ptr<uchar>();
	uchar* centre2_ptr = centre2.ptr<uchar>();

    for( i = 0; i < N; i++ )
    {
 //       dist[i] = hammingDist(data + step*i, data + step*centers[0], dims);
    	tdist2[i] = hammingDist(data + step*i, centre1_ptr, dims);
        dist[i] = std::min(hammingDist(data + step*i, centre2_ptr, dims), tdist2[i]);
        sum0 += dist[i];
    }

    for( k = 0; k < K-2; k++ )
    {
        double bestSum = DBL_MAX;
        int bestCenter = -1;

        for( j = 0; j < trials; j++ )
        {
            double p = (double)rng*sum0, s = 0;
            for( i = 0; i < N-1; i++ )
                if( (p -= dist[i]) <= 0 )
                    break;
            int ci = i;

            parallel_for_(Range(0, N),
                         KMeansBinaryPPDistanceComputer(tdist2, data, dist, dims, step, step*ci));
            for( i = 0; i < N; i++ )
            {
                s += tdist2[i];
            }

            if( s < bestSum )
            {
                bestSum = s;
                bestCenter = ci;
                std::swap(tdist, tdist2);
            }
        }
        centers[k] = bestCenter;
        sum0 = bestSum;
        std::swap(dist, tdist);
    }

    for( k = 0; k < K-2; k++ )
    {
        const uchar* src = data + step*centers[k];
        uchar* dst = _out_centers.ptr<uchar>(k);
        for( j = 0; j < dims; j++ )
            dst[j] = src[j];
    }
    uchar* dst1 = _out_centers.ptr<uchar>(K-2);
    uchar* dst2 = _out_centers.ptr<uchar>(K-1);
    for( j = 0; j < dims; j++ ) {
        dst1[j] = centre1_ptr[j];
        dst2[j] = centre2_ptr[j];
    }
}

class KMeansBinaryDistanceComputer : public ParallelLoopBody
{
public:
    KMeansBinaryDistanceComputer( double *_distances,
                            int *_labels,
                            const Mat& _data,
                            const Mat& _centers )
        : distances(_distances),
          labels(_labels),
          data(_data),
          centers(_centers)
    {
    }

    void operator()( const Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;
        const int K = centers.rows;
        const int dims = centers.cols;

        for( int i = begin; i<end; ++i)
        {
            const uchar *sample = data.ptr<uchar>(i);
            int k_best = 0;
            double min_dist = DBL_MAX;

            for( int k = 0; k < K; k++ )
            {
                const uchar* center = centers.ptr<uchar>(k);
                const double dist = hammingDist(sample, center, dims);

                if( min_dist > dist )
                {
                    min_dist = dist;
                    k_best = k;
                }
            }

            distances[i] = min_dist;
            labels[i] = k_best;
        }
    }

private:
    KMeansBinaryDistanceComputer& operator=(const KMeansBinaryDistanceComputer&); // to quiet MSVC

    double *distances;
    int *labels;
    const Mat& data;
    const Mat& centers;
};

double kmeansBinary( InputArray _data, int K,
                   InputOutputArray _bestLabels,
                   TermCriteria criteria, int attempts,
                   int flags, OutputArray _centers )
{
    const int SPP_TRIALS = 3;
    Mat data0 = _data.getMat();
    bool isrow = data0.rows == 1;
    int N = isrow ? data0.cols : data0.rows;
    int dims = (isrow ? 1 : data0.cols)*data0.channels();
    int type = data0.depth();

    attempts = std::max(attempts, 1);
    CV_Assert( data0.dims <= 2 && type == CV_8U && K > 0 );
    CV_Assert( N >= K );

    Mat data(N, dims, CV_8U, data0.ptr(), isrow ? dims * sizeof(uchar) : static_cast<size_t>(data0.step));

    _bestLabels.create(N, 1, CV_32S, -1, true);

    Mat _labels, best_labels = _bestLabels.getMat();
    if( flags & CV_KMEANS_USE_INITIAL_LABELS )
    {
        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                  best_labels.cols*best_labels.rows == N &&
                  best_labels.type() == CV_32S &&
                  best_labels.isContinuous());
        best_labels.copyTo(_labels);
    }
    else
    {
        if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
             best_labels.cols*best_labels.rows == N &&
            best_labels.type() == CV_32S &&
            best_labels.isContinuous()))
            best_labels.create(N, 1, CV_32S);
        _labels.create(best_labels.size(), best_labels.type());
    }
    int* labels = _labels.ptr<int>();

    Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type), votes(K, dims*8, CV_32S);
    std::vector<int> counters(K);
    double best_compactness = DBL_MAX, compactness = 0;
    RNG& rng = theRNG();
    int a, iter, i, j, k;

    if( criteria.type & TermCriteria::EPS )
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = FLT_EPSILON;
//    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
    else
        criteria.maxCount = 100;

    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }

    const uchar* sample = data.ptr<uchar>(0);

    for( a = 0; a < attempts; a++ )
    {
    	std::cout<<"attempt "<<a<<std::endl;
        double max_center_shift = DBL_MAX;
        for( iter = 0;; )
        {
        	std::cout<<"iteration "<<iter<<std::endl;
            swap(centers, old_centers);

//            std::cout<<"here1"<<std::endl;

            votes = Scalar(0);

 //           std::cout<<"here2"<<std::endl;

            if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
            {
                if( flags & KMEANS_PP_CENTERS ) {
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS);
//                    std::cout<<"init centers "<<std::endl;
//                    std::cout<<centers<<std::endl;
                }
                else
                {
                	CV_Assert(false);
//                    for( k = 0; k < K; k++ )
//                        generateRandomCenter(_box, centers.ptr<float>(k), rng);
                }
            }
            else
            {
                if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
                {
                    for( i = 0; i < N; i++ )
                        CV_Assert( (unsigned)labels[i] < (unsigned)K );
                }

                // compute centers
                centers = Scalar(0);
                for( k = 0; k < K; k++ )
                    counters[k] = 0;

//                std::cout<<"here3"<<std::endl;

                for( i = 0; i < N; i++ )
                {
                    sample = data.ptr<uchar>(i);
                    k = labels[i];
                    j=0;

//                    std::cout<<"here4"<<std::endl;


                    int* v = votes.ptr<int>(k);
                    for(int j = 0; j < dims; ++j, ++sample)
                    {
                      if(*sample & (1 << 7)) ++v[ j*8     ];
                      if(*sample & (1 << 6)) ++v[ j*8 + 1 ];
                      if(*sample & (1 << 5)) ++v[ j*8 + 2 ];
                      if(*sample & (1 << 4)) ++v[ j*8 + 3 ];
                      if(*sample & (1 << 3)) ++v[ j*8 + 4 ];
                      if(*sample & (1 << 2)) ++v[ j*8 + 5 ];
                      if(*sample & (1 << 1)) ++v[ j*8 + 6 ];
                      if(*sample & (1))      ++v[ j*8 + 7 ];
                    }

//                    std::cout<<"here5"<<std::endl;


                    counters[k]++;
                }

                if( iter > 0 )
                    max_center_shift = 0;


                for( k = 0; k < K; k++ )
                {
                    if( counters[k] != 0 )
                    {
//                        std::cout<<"here6"<<std::endl;

                   		int maj = (int)counters[k] / 2 + counters[k] % 2;
                        int* v = votes.ptr<int>(k);
                        uchar* c = centers.ptr<uchar>(k);

                   		for(i = 0; i < dims*8; ++i, ++v)
                   	    {
                   	      if(*v >= maj)
                   	      {
                   	        // set bit
                   	        *c |= 1 << (7 - (i % 8));
                   	      }

                   	      if(i % 8 == 7) ++c;
                   	    }
//                        std::cout<<"here7"<<std::endl;


                        if( iter > 0 )
                        {
                            double dist = 0;
                            const uchar* old_center = old_centers.ptr<uchar>(k);
                            const uchar* new_center = centers.ptr<uchar>(k);
                            dist = hammingDist(new_center, old_center, dims);
                            max_center_shift = std::max(max_center_shift, dist);
                        }
//                        std::cout<<"here8"<<std::endl;

                    }
                    else
                    {
						std::cout<<"a cluster is empty"<<std::endl;
						// if some cluster appeared to be empty then:
						//   1. find the biggest cluster
						//   2. find the farthest from the center point in the biggest cluster
						//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
						int max_k = 0;
						for( int k1 = 1; k1 < K; k1++ )
						{
							if( counters[max_k] < counters[k1] )
								max_k = k1;
						}

						double max_dist = 0;
						int farthest_i = -1;
						uchar* old_center = centers.ptr<uchar>(max_k);

						for( i = 0; i < N; i++ )
						{
							if( labels[i] != max_k )
								continue;
							sample = data.ptr<uchar>(i);
							double dist = hammingDist(sample, old_center, dims);

							if( max_dist <= dist )
							{
								max_dist = dist;
								farthest_i = i;
							}
						}

						counters[max_k]--;
						counters[k]++;
						labels[farthest_i] = k;

                    //skipping moving the point from the old to the new cluster
                    }
                }

//                std::cout<<"centers "<<std::endl;
//                std::cout<<centers<<std::endl;

            }

            std::cout<<"max_center_shift "<<max_center_shift<<std::endl;

            if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
                break;

            // assign labels
            Mat dists(1, N, CV_64F);
            double* dist = dists.ptr<double>(0);
            parallel_for_(Range(0, N),
                         KMeansBinaryDistanceComputer(dist, labels, data, centers));
//            std::cout<<"labels "<<std::endl;
//            std::cout<<_labels<<std::endl;
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                compactness += dist[i];
            }
        }

        std::cout<<"compactness "<<compactness<<std::endl;

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
            if( _centers.needed() )
                centers.copyTo(_centers);
            _labels.copyTo(best_labels);
        }
    }

    return best_compactness;
}

BoWKmeansppBinaryTrainer::BoWKmeansppBinaryTrainer( int _clusterCount, const cv::TermCriteria& _termcrit,
                                    int _attempts, int _flags ) :
		cv::BOWKMeansTrainer (_clusterCount, _termcrit, _attempts, _flags)
{}

cv::Mat BoWKmeansppBinaryTrainer::cluster( const cv::Mat& _descriptors ) const
{
    cv::Mat labels, vocabulary;
    kmeansBinary( _descriptors, clusterCount, labels, termcrit, attempts, flags, vocabulary );
    return vocabulary;
}

/*
void cluster(const Mat& descriptors) {
	//initialize clusters

	//loop until convergence or max iterations

	//compute mean

	//compute assignments

	//check for convergence

}
*/
}
