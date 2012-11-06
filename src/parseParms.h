#ifndef PARSEPARMS_H
#define PARSEPARMS_H

#include "dataType.h"
#include <mex.h>
#include <string>
#include <vector>
#include <opencv/cxcore.h>
#include "utility.h"
#include <omp.h>

typedef struct 
{
	mxArray *p;
	double *data;
	double h;
	double w;
	double d;
	double arraySize;
}dataMap ;


class patchMatch{
public:	
	//	
	dataMap _depthMaps;
	dataMap _depthRandomMaps;
	dataMap _distributionMap;
	dataMap _orientationMap;

	std::vector<ImageStruct> _imgStruct_1;
	std::vector<ImageStruct> _imgStruct_2;
	double _halfWindowSize;
	//double _near;
	//double _far;
	int _numOfSamples;
	int _numOfThreadsUsed;

	double _sigma;
	timer _tt;

	//double _totalTime;

	patchMatch(const mxArray *prhs[])
	{
		
		parseImageStructure(&_imgStruct_1, prhs[0]);		// image structure 1
		parseImageStructure(&_imgStruct_2, prhs[1]);		// image structure 2
		for(int i = 0; i<_imgStruct_2.size(); i++)
		{
			_imgStruct_2[i].init_relative(_imgStruct_1[0]);
		}

		parseDataMap(&_depthMaps, prhs[2]);
		parseDataMap(&_depthRandomMaps, prhs[3]);
		parseDataMap(&_distributionMap, prhs[4]);
		parseDataMap(&_orientationMap, prhs[6]);
		_halfWindowSize = 4;
		//_near = 3;
		//_far = 15;
		_numOfSamples = 5;
		_sigma = 0.2;
		_numOfThreadsUsed = omp_get_max_threads() - 9;

		//_totalTime = 0;
	}

	void parseImageStructure(std::vector<ImageStruct> *allImgStruct, const mxArray *p);
	void parseDataMap(dataMap *dataMaps, const mxArray *p);
	//pixelColor fetchColorOnePixel(const double *imageData, const int &ind_r, const int &ind_g, const int &ind_b );

	void leftToRight();
	void RightToLeft();
	void TopToDown();
	void DownToTop();

	inline void findRange(const double &row, const double &col, double &rowStart, double &rowEnd, double &colStart, double &colEnd, const double &halfWindowSize, const double &w, const double &h)
	{
		rowStart = row - halfWindowSize >= 0 ? (row - halfWindowSize) : 0;
		rowEnd = row + halfWindowSize <= h-1 ? (row + halfWindowSize) : h-1;
		colStart = col - halfWindowSize >= 0 ? (col - halfWindowSize) : 0;
		colEnd = col + halfWindowSize <= w-1 ? (col + halfWindowSize) : w-1;
	}
	
	void findPixelPos(pixelPos *pixelPostions, const int &numOfPixels, const double &rowStart, const double &rowEnd, const double &colStart, const double &colEnd);
	void findPixelColors(std::vector<pixelColor> &pColors, pixelPos *pPos, ImageStruct &img, const int &numOfPixels);

	void findPixelColorsInterpolation(std::vector<pixelColor> &pColors, const std::vector<pixelPos> &pPos, ImageStruct &img, const int& numOfPixels);

	void updateDistribution(const pixelPos &formerPixel, const pixelPos &currentPixel, dataMap& distributionMap);

	void drawSamples(const std::vector<double> &distribution, int numOfSamples, std::vector<int> &samples);
	
	void drawSamples(const dataMap &distributionMap, std::vector<int> &imageLayerId, int numOfSamples, const pixelPos &curPixel);
	void normalizeDistribution(std::vector<double> &distribution);

	void computeCost(double &cost, const std::vector<pixelColor> &refPixelColor, const pixelPos* refPixelPos, const std::vector<pixelColor> &refNormColor,  const pixelColor &deviationColor,
		int imageId, const double &depth,  const int& numOfPixels, std::vector<pixelPos> &otherImagePixelPos, std::vector<pixelColor> &otherImagePixelColor, const cv::Mat &orientation);

	void getOtherImagePixelPos(std::vector<pixelPos> &otherImagePixelPos, /*const std::vector<pixelPos> &*/const pixelPos* refPixelPos, 
		double depth, int imageId, const int& numOfPixels, const cv::Mat &orientation);
	
	double calculateNCC(std::vector<pixelColor> &otherImagePixelColor, const std::vector<pixelColor> &refPixelColor, const int &numOfPixels);
	void UpdateDistributionMap(const std::vector<double> &cost, const pixelPos &currentPos, const dataMap & distributionMap, std::vector<double> &prob);
	int findBestDepth_average(const std::vector<double> &cost, std::vector<bool> &testedIdSet);
	pixelColor meanNormalize(const std::vector<pixelColor> &refPixelColor, std::vector<pixelColor> &refNormColor, const int &numOfPixels);

	bool determineFullPatch(const std::vector<pixelColor> &otherImagePixelColor, const int &numOfPixels);

	double calculateNCC_withNormalized(const std::vector<pixelColor> &refNormColor, const pixelColor &refDeviationColor, 
		const std::vector<pixelColor> &otherNormColor, const pixelColor &otherImageDeviationColor, const int &numOfPixels);

	void getOrientation(const dataMap &orientationMap, int pixelIdx, cv::Mat &orientation);
	void getRandomOrientation(cv::Mat &orientation);
	void assignOrientationMap(dataMap &orientationMap, const int &currentPixelIdx, const cv::Mat &orientation);

	void wrap1(const double &row, const double &col, double &colStart, double &colEnd,
		double &rowStart, double &rowEnd, int &numOfPixels, pixelPos *refPixelPos, std::vector<pixelColor> &refPixelColor, const double &ref_w, const double &ref_h);

	void patchMatch::wrap2(int &formerPixelIdx, int &currentPixelIdx, double *depth, 
		cv::Mat *orientation, std::vector<int> *imageLayerId, const int &numOfPixels,
		pixelPos &formerPixel, pixelPos &currentPixel, 
		std::vector<pixelColor> &refPixelColor, std::vector<pixelColor> &refNormColor, pixelPos *refPixelPos,
		std::vector<double> &cost,  std::vector<bool> &testedIdSet, int &bestDepthId,
		std::vector<double> &costWithBestDepth, std::vector<pixelPos> &otherImagePixelPos, std::vector<pixelColor> &otherImagePixelColor, std::vector<double> &prob
	);

	int findBestDepth_votes(const std::vector<double> &cost, std::vector<bool> &testedIdSet);
	void drawSamples_average(const dataMap &distributionMap, std::vector<int> &imageLayerId, int numOfSamples, const pixelPos &curPixel, const pixelPos &formerPixel);
};

#endif