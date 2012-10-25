#ifndef PARSEPARMS_H
#define PARSEPARMS_H

#include "dataType.h"
#include <mex.h>
#include <string>
#include <vector>
#include <opencv/cxcore.h>

typedef struct 
{
	mxArray *p;
	double *data;
	double h;
	double w;
	double d;
}dataMap ;


class patchMatch{
public:	
	//	
	dataMap _depthMaps;
	dataMap _depthRandomMaps;
	dataMap _distributionMap;
	std::vector<ImageStruct> _imgStruct_1;
	std::vector<ImageStruct> _imgStruct_2;
	double _halfWindowSize;
	double _near;
	double _far;
	int _numOfSamples;
	double _sigma;

	patchMatch(const mxArray *prhs[])
	{
		parseImageStructure(&_imgStruct_1, prhs[0]);		// image structure 1
		parseImageStructure(&_imgStruct_2, prhs[1]);		// image structure 2
		parseDataMap(&_depthMaps, prhs[2]);		
		parseDataMap(&_depthRandomMaps, prhs[3]);
		parseDataMap(&_distributionMap, prhs[4]);
		_halfWindowSize = 4;
		_near = 3;
		_far = 12;
		_numOfSamples = 3;
		_sigma = 0.2;
	}

	void parseImageStructure(std::vector<ImageStruct> *allImgStruct, const mxArray *p);
	void parseDataMap(dataMap *dataMaps, const mxArray *p);
	pixelColor fetchColorOnePixel(const double *imageData, const int &ind_r, const int &ind_g, const int &ind_b );

	void leftToRight();
	void findRange(const double &row, const double &col, double &rowStart, double &rowEnd, double &colStart, double &colEnd, const double &halfWindowSize, const double &w, const double &h);
	void findPixelPos(std::vector<pixelPos> &pixelPos , const double &rowStart, const double &rowEnd, const double &colStart, const double &colEnd);
	void findPixelColors(std::vector<pixelColor> &pColors, const std::vector<pixelPos> &pPos, ImageStruct &img);

	void findPixelColorsInterpolation(std::vector<pixelColor> &pColors, const std::vector<pixelPos> &pPos, ImageStruct &img);

	void updateDistribution(const pixelPos &formerPixel, const pixelPos &currentPixel, dataMap& distributionMap);

	void drawSamples(const std::vector<double> &distribution, int numOfSamples, std::vector<int> &samples);
	
	void drawSamples(const dataMap &distributionMap, std::vector<int> &imageLayerId, int numOfSamples, const pixelPos &curPixel);
	void normalizeDistribution(std::vector<double> &distribution);

	void computeCost(double &cost, const std::vector<pixelColor> &refPixelColor, 
		const std::vector<pixelPos> &refPixelPos, int imageId, double depth);
	void getOtherImagePixelPos(std::vector<pixelPos> &otherImagePixelPos, const std::vector<pixelPos> &refPixelPos, double depth, int imageId);
	double calculateNCC(const std::vector<pixelColor> &otherImagePixelColor, const std::vector<pixelColor> &refPixelColor);
	void UpdateDistributionMap(const std::vector<double> &cost, const pixelPos &currentPos, const dataMap & distributionMap);
	int findBestDepth_average(const std::vector<double> &cost, std::vector<bool> &testedIdSet);

};

#endif