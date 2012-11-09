#include "parseParms.h"
#include <assert.h>
#include "utility.h"
#include <omp.h>
#include <math.h>

#define USE_OPENMP

#define UNSET -2.0

void patchMatch:: parseImageStructure(std::vector<ImageStruct> *allImgStruct, const mxArray* prhs)
{
	int nfields = mxGetNumberOfFields(prhs);	
	size_t NStructElems = mxGetNumberOfElements(prhs);	
	for(int i = 0; i< NStructElems; i++)
	{
		std::vector<std::string> fnames;
		mxArray *tmp;
		ImageStruct imgStruct;
		for (int ifield=0; ifield< nfields; ifield++)
		{
			fnames.push_back(std::string(mxGetFieldNameByNumber(prhs,ifield)));	
			tmp = mxGetFieldByNumber(prhs, i, ifield);	

			if(fnames[ifield] == "K")			
			{				
				imgStruct.K = mxGetPr(tmp);				
			}
			else if(fnames[ifield] == "R")
			{
				imgStruct.R = mxGetPr(tmp);				
			}
			else if(fnames[ifield] == "T")
			{
				imgStruct.T = mxGetPr(tmp);
			}
			else if(fnames[ifield] == "C")
			{
				imgStruct.C = mxGetPr(tmp);
			}
			else if(fnames[ifield] == "imageName")
			{
				;
			}	
			else if(fnames[ifield] == "imageData")
			{
				imgStruct.imageData = mxGetPr(tmp);
			}
			else if(fnames[ifield] == "h")
			{
				imgStruct.h = *mxGetPr(tmp);
			}
			else if(fnames[ifield] == "w")
			{
				imgStruct.w = *mxGetPr(tmp);
			}
			else if(fnames[ifield] == "d")
			{
				imgStruct.d = *mxGetPr(tmp);
			}			
		}
		imgStruct.init();
		allImgStruct->push_back(imgStruct);
	}
}

void patchMatch::parseDataMap(dataMap *dataMaps, const mxArray *p)
{
	dataMaps->p = mxDuplicateArray(p);		// deep copy of the array
	dataMaps->data = mxGetPr(dataMaps->p);	// this pointer points to copied array
	mwSize numOfDimensions = mxGetNumberOfDimensions(p);

	const mwSize *sz = mxGetDimensions(dataMaps->p);
	dataMaps->h = static_cast<double>(sz[0]);
	dataMaps->w = static_cast<double>(sz[1]);
	if(numOfDimensions == 3)
		dataMaps->d = static_cast<double>(sz[2]);
	else
		dataMaps->d = 1;

	dataMaps->arraySize = dataMaps->h * dataMaps->w;
	
}

//void patchMatch::findRange(const double &row, const double &col, double &rowStart, double &rowEnd, double &colStart, double &colEnd, const double &halfWindowSize, const double &w, const double &h)
//{
//	rowStart = row - halfWindowSize >= 0 ? (row - halfWindowSize) : 0;
//	rowEnd = row + halfWindowSize <= h-1 ? (row + halfWindowSize) : h-1;
//	colStart = col - halfWindowSize >= 0 ? (col - halfWindowSize) : 0;
//	colEnd = col + halfWindowSize <= w-1 ? (col + halfWindowSize) : w-1;
//}

void patchMatch::findPixelPos(pixelPos *pixelPostions, const int &numOfPixels, const double &rowStart, const double &rowEnd, const double &colStart, const double &colEnd)
{
	//int numOfPixels = static_cast<int>((rowEnd - rowStart +1) * (colEnd - colStart + 1));
	//pixelPostions.reserve(numOfPixels);
	int count = 0;
	for(double i = rowStart; i<=rowEnd; i++)
	{
		for(double j = colStart; j<=colEnd; j++)
		{
			//pixelPostions.push_back(pixelPos(j, i));			
			pixelPostions[count]._pt.at<double>(0) = j;
			pixelPostions[count]._pt.at<double>(1) = i;
			count++;
		}
	}
}

void patchMatch::findPixelColors(std::vector<pixelColor> &pColors, pixelPos *pPos, ImageStruct &img, const int &numOfPixels)
{
	//size_t numOfPixels = pPos.size();
	//for(int i = 0; i < numOfPixels; i++)
	//{
	//	double ind_r, ind_g, ind_b;
	//	pPos[i].sub2Idx(ind_r, ind_g, ind_b, img.h, img.arraySize);
	//	//pColors.push_back( fetchColorOnePixel(img.imageData, static_cast<int>(ind_r), static_cast<int>(ind_g), static_cast<int>(ind_b)));
	//	//pColors.push_back( pixelColor(img.imageData[static_cast<int>(ind_r)], img.imageData[static_cast<int>(ind_g)], img.imageData[static_cast<int>(ind_b)]) );
	//	//pColors.push_back( pixelColor(img.imageData[static_cast<int>(ind_r)], img.imageData[static_cast<int>(ind_g)], img.imageData[static_cast<int>(ind_b)]) );
	//	pColors[i]._color.at<double>(0) = img.imageData[static_cast<int>(ind_r)];
	//	pColors[i]._color.at<double>(1) = img.imageData[static_cast<int>(ind_g)];
	//	pColors[i]._color.at<double>(2) = img.imageData[static_cast<int>(ind_b)];
	//	pColors[i]._color.at<double>(3) = 0.0;
	//	//imageData[ind_r], imageData[ind_g], imageData[ind_b]
	//}

	//double offset[3] = {0, img.arraySize, 2 * img.arraySize};
	for(int chan = 0; chan < 3; chan++)
	{
		//
		for(int i = 0; i < numOfPixels; i++)
		{
			double ind = pPos[i].sub2Idx(img.h) + chan * img.arraySize; 
			//double ind = pPos[i].sub2Idx(img.h) + offset[chan]; 
			pColors[i]._color.at<double>(chan) = img.imageData[static_cast<int>(ind)];
		}
	}	

}

void patchMatch::findPixelColorsInterpolation(std::vector<pixelColor> &pColors, const std::vector<pixelPos> &pPos, ImageStruct &img, const int& numOfPixels)
{
	//size_t numOfPixels = pPos.size();
	/*std::vector<double> ind_r;
	std::vector<double> ind_g;
	std::vector<double> ind_b;
	std::vector<double> weight;*/
	double ind_r[4];
	double ind_g[4];
	double ind_b[4];
	double weight[4];

	for(int i = 0; i < numOfPixels; i++)
	{
		//pixelColor pCol(0.0, 0.0, 0.0);
		for(int j = 0; j<4; j++)
			pColors[i]._color.at<double>(j) = 0.0;	// initialize

		if( pPos[i]._pt.at<double>(0) < 0 || pPos[i]._pt.at<double>(0) >= img.w - 1.0 ||
			pPos[i]._pt.at<double>(1) < 0 || pPos[i]._pt.at<double>(1) >= img.h - 1.0 )
		{
			//pCol._color.at<double>(3) = 1.0;	
			//pColors[i]._color.at<double>(0) = 0.0;
			//pColors[i]._color.at<double>(1) = 0.0;
			//pColors[i]._color.at<double>(2) = 0.0;
			pColors[i]._color.at<double>(3) = 1.0;
		}
		else
		{
			pPos[i].sub2Idx(weight, ind_r, ind_g, ind_b, img.h, img.arraySize);				
			double g[4];
			for(int j = 0; j<4; j++)
			{		
				g[j] = img.imageData[static_cast<int>(ind_g[j])];
				pColors[i]._color.at<double>(0) += (img.imageData[static_cast<int>(ind_r[j])] * weight[j]);
				pColors[i]._color.at<double>(1) += (img.imageData[static_cast<int>(ind_g[j])] * weight[j]);
				pColors[i]._color.at<double>(2) += (img.imageData[static_cast<int>(ind_b[j])] * weight[j]);
				//pColors[i]._color.at<double>(3) = 0.0f;		
			}
		}			
	}
}

//pixelColor patchMatch::fetchColorOnePixel(const double *imageData, const int &ind_r, const int &ind_g, const int &ind_b )
//{
//	double a = imageData[ind_r];
//	double b = imageData[ind_g];
//	double c = imageData[ind_b];
//	return pixelColor(imageData[ind_r], imageData[ind_g], imageData[ind_b]);		
//	
//}

void patchMatch::updateDistribution(const pixelPos &formerPixel, const pixelPos &currentPixel, dataMap& distributionMap)
{	
	// update distribution for each layer
	for(int layer = 0; layer < distributionMap.d; layer++)
	{
		double curIdx = currentPixel.sub2Idx(distributionMap.h, distributionMap.arraySize, layer );
		double formerIdx = formerPixel.sub2Idx(distributionMap.h, distributionMap.arraySize, layer);
		distributionMap.data[static_cast<int>(curIdx)] += distributionMap.data[static_cast<int>(formerIdx)];
		distributionMap.data[static_cast<int>(curIdx)] /= 2.0f;
	}
}

void patchMatch::normalizeDistribution(std::vector<double> &distribution)
{
	double sum = 0;
	for(int i = 0; i<distribution.size(); i++)
	{
		sum += distribution[i];
	}
	for(int i = 0; i<distribution.size(); i++)
	{
		distribution[i] /= sum;
	}
}

void patchMatch::drawSamples(const std::vector<double> &distribution, int numOfSamples, std::vector<int>& samples)
{	
	
	std::vector<double> accumulated; 
	accumulated.resize(distribution.size());
	accumulated[0] = distribution[0];
	for(int i = 1; i < distribution.size(); i++)
	{
		accumulated[i] = accumulated[i-1] + distribution[i];
	}
	// draw samples based on distribution
	samples.clear();	
	for(int i = 0; i<numOfSamples; i++)
	{
		double rdNum = rand()/static_cast<double>(RAND_MAX);
		for(int j = 0; j<distribution.size(); j++)
		{
			if(rdNum < accumulated[j])
			{
				bool isUnique = true;
				for(int k = 0; k < samples.size(); k++)
				{
					if(j == samples[k])
					{
						isUnique = false;
						break;
					}
				}
				if(isUnique)
					samples.push_back(j);

				break;
			}
		}
	}

}

void patchMatch::drawSamples(const dataMap &distributionMap, std::vector<int> &imageLayerId, int numOfSamples, const pixelPos &curPixel)
{
	
	std::vector<double> distribution;
	distribution.resize(static_cast<int>(distributionMap.d));
	for(int i = 0; i<distribution.size(); i++)
	{
		distribution[i] = _distributionMap.data[static_cast<int>( curPixel.sub2Idx(distributionMap.h, distributionMap.arraySize, i))];
	}
	// normalize distribution:
	normalizeDistribution(distribution);
	drawSamples(distribution, numOfSamples, imageLayerId);
}

void patchMatch::drawSamples_average(const dataMap &distributionMap, std::vector<int> &imageLayerId, int numOfSamples, const pixelPos &curPixel, const pixelPos &formerPixel)
{
	std::vector<double> distribution;
	distribution.resize(static_cast<int>(distributionMap.d));
	for(int i = 0; i<distribution.size(); i++)
	{
		distribution[i] = (_distributionMap.data[static_cast<int>( curPixel.sub2Idx(distributionMap.h, distributionMap.arraySize, i))]  +
			_distributionMap.data[static_cast<int>( formerPixel.sub2Idx(distributionMap.h, distributionMap.arraySize, i))]) * 0.5;

	}
	// normalize distribution:
	normalizeDistribution(distribution);
	drawSamples(distribution, numOfSamples, imageLayerId);

}


bool patchMatch::determineFullPatch(const std::vector<pixelColor> &otherImagePixelColor, const int &numOfPixels)
{
	for(int i = 0; i < numOfPixels; i++)
	{
		if(otherImagePixelColor[i]._color.at<double>(3) == 1.0f)
		{
			return false;
		}		
	}
	return true;
}

double patchMatch :: calculateNCC_withNormalized(const std::vector<pixelColor> &refNormColor, const pixelColor &refDeviationColor, 
	const std::vector<pixelColor> &otherNormColor, const pixelColor &otherImageDeviationColor, const int &numOfPixels)
{
//	refNormColor, refDeviationColor, otherImagePixelColor, otherImageDeviationColor

	/*double a1 = 0,a2 = 0;
	for(int i = 0; i<refNormColor.size(); i++)
	{
		a1 += refNormColor[i]._color.at<double>(0) * refNormColor[i]._color.at<double>(0);
		a2 += otherNormColor[i]._color.at<double>(0) * otherNormColor[i]._color.at<double>(0);
	}*/



	double cost[3] = {0};
	for(int i = 0; i < numOfPixels; i++)
	{
		for( int j = 0; j<3; j++)
			cost[j] += refNormColor[i]._color.at<double>(j) * otherNormColor[i]._color.at<double>(j);		
	}
	for(int i = 0; i< 3; i++)
		cost[i] /= (refDeviationColor._color.at<double>(i) * otherImageDeviationColor._color.at<double>(i) + 0.000000000000001);	// devide by 0


	return (cost[0] + cost[1] + cost[2])/3.0;
}


void patchMatch:: computeCost(double &cost, const std::vector<pixelColor> &refPixelColor, const pixelPos* refPixelPos, const std::vector<pixelColor> &refNormColor, const pixelColor &refDeviationColor,
	int imageId, const double &depth, const int& numOfPixels, std::vector<pixelPos> &otherImagePixelPos, std::vector<pixelColor> &otherImagePixelColor, const cv::Mat &orientation)
{	
	
	getOtherImagePixelPos(otherImagePixelPos, refPixelPos, depth, imageId, numOfPixels, orientation);
	
	//_tt.startTimer();
	findPixelColorsInterpolation(otherImagePixelColor, otherImagePixelPos, _imgStruct_2[imageId], numOfPixels);	// 	
	//_tt.calculateTotalTime();

	_tt.startTimer();
	bool isFullPatch = determineFullPatch(otherImagePixelColor, numOfPixels);

	if(isFullPatch)
	{// use refNormColor
		//std::vector<pixelColor> otherNormColor;
		//otherNormColor.resize(numOfPixels);
		pixelColor otherImageDeviationColor = meanNormalize(otherImagePixelColor, otherImagePixelColor, numOfPixels);
		cost = calculateNCC_withNormalized(refNormColor, refDeviationColor, otherImagePixelColor, otherImageDeviationColor, numOfPixels);		
	}
	else
	{
		//_tt.startTimer();
		cost = calculateNCC( otherImagePixelColor, refPixelColor, numOfPixels);
		//_tt.calculateTotalTime();
	}
	_tt.calculateTotalTime();
		
	
}	

void patchMatch::
	(std::vector<pixelPos> &otherImagePixelPos, /*const std::vector<pixelPos>*/ const pixelPos* refPixelPos, double depth, int imageId, const int& numOfPixels, const cv::Mat &orientation)
{
	//cv::Mat normalVector = (cv::Mat_<double>(1,3) << 0, 0, 1);
	//cv::Mat opencv_R = _imgStruct_2[imageId].opencv_R *_imgStruct_1[0].opencv_inverseR ;
	//cv::Mat opencv_T = _imgStruct_2[imageId].opencv_R * (_imgStruct_2[imageId].opencv_C - _imgStruct_1[0].opencv_C);	
	//cv::Mat H = _imgStruct_2[imageId].opencv_K * (opencv_R - opencv_T * normalVector / depth) * _imgStruct_1[0].opencv_inverseK;
	//cv::Mat H = _imgStruct_2[imageId].opencv_K * (_imgStruct_2[imageId].opencv_relative_R - _imgStruct_2[imageId].opencv_relative_T * normalVector / depth) * _imgStruct_1[0].opencv_inverseK;
	
	cv::Mat H2 = _imgStruct_2[imageId].opencv_K * _imgStruct_2[imageId].opencv_relative_T * orientation * _imgStruct_1[0].opencv_inverseK;
	
	//cv::Mat pixelIn3dCoord = _imgStruct_1[0].opencv_inverseK

	cv::Mat H = _imgStruct_2[imageId].H1 - (H2/depth);
	

//		H2 = opencv_K * opencv_relative_T * normalVector * refImg.opencv_inverseK;


	//int numOfPixels = static_cast<int>(refPixelPos.size());
	
	//otherImagePixelPos.resize(numOfPixels);
	for(int i = 0; i<numOfPixels; i++)
	{
		//pixelPos refImageOnePixel(refPixelPos[i]._pt.at<double>(0) + 0.5, refPixelPos[i]._pt.at<double>(1) + 0.5 );
		//pixelPos otherImageOnePixel(refPixelPos[i]._pt.at<double>(0) + 0.5, refPixelPos[i]._pt.at<double>(1) + 0.5 );
		otherImagePixelPos[i]._pt.at<double>(0) = refPixelPos[i]._pt.at<double>(0) + 0.5;
		otherImagePixelPos[i]._pt.at<double>(1) = refPixelPos[i]._pt.at<double>(1) + 0.5;
		otherImagePixelPos[i]._pt = H * otherImagePixelPos[i]._pt;
		otherImagePixelPos[i]._pt /= otherImagePixelPos[i]._pt.at<double>(2); // normalize
		// normalize:
		//newPixel = newPixel/newPixel.at<double>(2);
		//otherImagePixelPos.push_back(pixelPos(newPixel)); 
		//otherImagePixelPos.push_back(otherImageOnePixel);
	}

}


double patchMatch::calculateNCC(std::vector<pixelColor> &otherImagePixelColor, const std::vector<pixelColor> &refPixelColor, const int& numOfPixels)
{
	//assert(otherImagePixelColor.size() == refPixelColor.size());
	std::vector<pixelColor> refPixelTempColor;
	refPixelTempColor.resize(refPixelColor.size());
	
	pixelColor otherImageMean(0., 0., 0.);
	pixelColor refImageMean(0.,0.,0.);	
	int numOfValidPixels = 0;

	for(int i = 0; i<numOfPixels; i++)
	{
		if(otherImagePixelColor[i]._color.at<double>(3) != 1.0f)
		{
			++numOfValidPixels;
		}
	}
	if(numOfValidPixels == 0)
	{
		return -1;	
	}

	for(int i = 0; i < numOfPixels; i++)
	{
		if(otherImagePixelColor[i]._color.at<double>(3) != 1.0f)
		{
			otherImageMean._color += otherImagePixelColor[i]._color;
			refImageMean._color += refPixelColor[i]._color;
			//numOfValidPixels += 1;			
		}		
	}
	
	//otherImageMean = otherImageMean * (1/ static_cast<double>(numOfValidPixels));
	//refImageMean = refImageMean * (1/ static_cast<double>(numOfPixels));
	double inverseNumOfValidPixels = 1.0/static_cast<double>(numOfValidPixels);

	otherImageMean._color *= inverseNumOfValidPixels;
	refImageMean._color *= inverseNumOfValidPixels;

	//otherImageMean._color /= static_cast<double>(numOfValidPixels);
	//refImageMean._color /= static_cast<double>(numOfValidPixels);

	// 
	//std::vector<pixelColor> otherImageSubtracted;
	//std::vector<pixelColor> refImageSubtracted;
	pixelColor otherImageSigma(0,0,0);
	pixelColor refImageSigma(0,0,0);
	for(int i= 0; i < numOfPixels; i++ )
	{
		if(otherImagePixelColor[i]._color.at<double>(3) != 1.0f)
		{
			//otherImageSubtracted.push_back( otherImagePixelColor[i] - otherImageMean );
			//refImageSubtracted.push_back( refPixelColor[i] - refImageMean ); 
			otherImagePixelColor[i]._color -= otherImageMean._color;
			//refPixelColor[i]._color -= refImageMean._color;
			refPixelTempColor[i]._color = refPixelColor[i]._color - refImageMean._color;

			//this->_color.mul(p._color);
			otherImageSigma._color += otherImagePixelColor[i]._color.mul(otherImagePixelColor[i]._color);
			refImageSigma._color += refPixelTempColor[i]._color.mul(refPixelTempColor[i]._color);

			//otherImageSigma._color += (otherImagePixelColor[i] * otherImagePixelColor[i])._color;
			//refImageSigma._color += (refPixelColor[i] * refPixelColor[i])._color;
		}
	}
	//
	
	/*for(int i = 0; i< numOfValidPixels; i++)
	{
		otherImageSigma += (otherImageSubtracted[i] * otherImageSubtracted[i]);
		refImageSigma += (refImageSubtracted[i] * refImageSubtracted[i]);
	}*/
	
	//otherImageSigma._color /= static_cast<double>(numOfValidPixels);
	otherImageSigma._color *= inverseNumOfValidPixels;
	otherImageSigma.sqrtRoot();
	//refImageSigma._color /= static_cast<double>(numOfValidPixels);	
	refImageSigma._color *= inverseNumOfValidPixels;
	refImageSigma.sqrtRoot();

	double cost_rgb[3] = {0};
	//
	
	for(int j = 0; j< numOfPixels; j++)
	{
		if(otherImagePixelColor[j]._color.at<double>(3) != 1.0f)
		{
			for(int i = 0; i<3; i++)
				//cost_rgb[i] += (otherImagePixelColor[j]._color.at<double>(i) * refPixelColor[j]._color.at<double>(i) / (otherImageSigma._color.at<double>(i) * refImageSigma._color.at<double>(i)+0.000000000001  ));
				cost_rgb[i] += (otherImagePixelColor[j]._color.at<double>(i) * refPixelTempColor[j]._color.at<double>(i));
		}
	}
	for(int i = 0; i<3; i++)
		 cost_rgb[i] /= (otherImageSigma._color.at<double>(i) * refImageSigma._color.at<double>(i)+0.000000000001  );


	//double cost = (cost_rgb[0] + cost_rgb[1] + cost_rgb[2])/3.0f/static_cast<double>(numOfValidPixels);
	//double cost = (cost_rgb[0] + cost_rgb[1] + cost_rgb[2])/3.0 * inverseNumOfValidPixels;
	//return cost;

	return (cost_rgb[0] + cost_rgb[1] + cost_rgb[2])/3.0 * inverseNumOfValidPixels;
}

void patchMatch::getOrientation(const dataMap &orientationMap, int pixelIdx, cv::Mat &orientation)
{
	orientation.at<double>(0) = orientationMap.data[pixelIdx];
	orientation.at<double>(1) = orientationMap.data[pixelIdx + static_cast<int>(orientationMap.arraySize)];
	orientation.at<double>(2) = orientationMap.data[pixelIdx + static_cast<int>(orientationMap.arraySize) * 2];
	if(orientation.at<double>(2) < 0)
	{
		orientation *= -1;
	}
}

void patchMatch::getRandomOrientation(cv::Mat &orientation)
{
	/*for(int i = 0; i<3; i++)
		orientation.at<double>(i) = rand()/static_cast<double>(RAND_MAX) * 2.0 - 1.0;	
	cv::normalize(orientation, orientation);
	if(orientation.at<double>(2) < 0)
	{
		orientation *= -1;
	}*/

	orientation.at<double>(0) = 0.0;
	orientation.at<double>(1) = 0.0;
	orientation.at<double>(2) = 1.0;

}

void patchMatch::wrap1(const double &row, const double &col, double &colStart, double &colEnd,
		double &rowStart, double &rowEnd, int &numOfPixels, pixelPos *refPixelPos, std::vector<pixelColor> &refPixelColor, 
		const double &ref_w, const double &ref_h)
{
	//1) find the start and end of the row and col (start smaller than end)
	findRange( row, col, rowStart, rowEnd, colStart, colEnd, _halfWindowSize, ref_w, ref_h);	

	//2)						
	numOfPixels = static_cast<int>(((colEnd - colStart + 1) * (rowEnd - rowStart + 1))); 
	findPixelPos(refPixelPos, numOfPixels ,rowStart, rowEnd, colStart, colEnd);	// 
			
	//3) find the color in reference image given pixels			
	//std::vector<pixelColor> refPixelColor;
	//refPixelColor.reserve(numOfPixels);
	findPixelColors(refPixelColor, refPixelPos, _imgStruct_1[0], numOfPixels);	// within this function, it should allow non-integer pixel positions
}

void patchMatch::wrap2(int &formerPixelIdx, int &currentPixelIdx, double *depth, 
	cv::Mat *orientation, std::vector<int> *imageLayerId, const int &numOfPixels,
	pixelPos &formerPixel, pixelPos &currentPixel, 
	std::vector<pixelColor> &refPixelColor, std::vector<pixelColor> &refNormColor, pixelPos *refPixelPos,
	std::vector<double> &cost,  std::vector<bool> &testedIdSet, int &bestDepthId, 
	std::vector<double> &costWithBestDepth, std::vector<pixelPos> &otherImagePixelPos, std::vector<pixelColor> &otherImagePixelColor, 
	std::vector<double> &prob
	)
{
		formerPixelIdx = static_cast<int>(formerPixel.sub2Idx(_depthMaps.h));
		currentPixelIdx = static_cast<int>(currentPixel.sub2Idx(_depthMaps.h));
						
		depth[0] = _depthMaps.data[formerPixelIdx]; 
		getOrientation(_orientationMap, formerPixelIdx, orientation[0]);
		depth[1] = _depthRandomMaps.data[currentPixelIdx]; /* randomly initialize a orientation*/ 
		getRandomOrientation(orientation[1]);
		depth[2] = _depthMaps.data[currentPixelIdx]; 
		getOrientation(_orientationMap, currentPixelIdx, orientation[2]);

		//5) draw samples and update the image ID distribution
		// draw samples:
		if(_numOfSamples == 1)
			drawSamples(_distributionMap, imageLayerId[0], _numOfSamples, formerPixel);			
		//drawSamples(_distributionMap, imageLayerId[1], _numOfSamples, currentPixel);	
		else
			//drawSamples_average(_distributionMap, imageLayerId[0], _numOfSamples, currentPixel, formerPixel);
			drawSamples(_distributionMap, imageLayerId[0], _numOfSamples, currentPixel);	
		imageLayerId[1] = imageLayerId[0];

		//imageLayerId[0].clear(); imageLayerId[0].push_back(1); imageLayerId[0].push_back(2);
		//imageLayerId[1].clear(); imageLayerId[1].push_back(3);

		//6) transforming the pixels to the other image (depth given, image id is given) and find the color for given pixels. calculate costs	
		std::fill(cost.begin(), cost.end(), UNSET);

		// calculate
		pixelColor deviationColor = meanNormalize(refPixelColor, refNormColor, numOfPixels);
		for(int j = 0; j < 2; j++ )	
		{
			for(int k = 0; k < imageLayerId[j].size(); k++ )				
			{
				if(cost[3 * imageLayerId[j][k]] == UNSET)
				{
					// I can calculate all the cost at the same time, in order to save time.
					for(int i = 0; i<3; i++)
					{	
						if( i == 0)
							computeCost(cost[i + 3 * imageLayerId[j][k]], refPixelColor, refPixelPos, refNormColor, deviationColor, imageLayerId[j][k], depth[i], numOfPixels, otherImagePixelPos, otherImagePixelColor, orientation[i]); // cost is the output							
						else
						{
							if(depth[i] == depth[0])
							{
								cost[i + 3 * imageLayerId[j][k]] = cost[0 + 3 * imageLayerId[j][k]];
							}
							else
								computeCost(cost[i + 3 * imageLayerId[j][k]], refPixelColor, refPixelPos, refNormColor, deviationColor, imageLayerId[j][k], depth[i], numOfPixels, otherImagePixelPos, otherImagePixelColor, orientation[i]); // cost is the output							
						}
					}
				}				
			}
		}
		//7) based on the cost, VOTE which depth to use. and then save the depth			
		bestDepthId = findBestDepth_average(cost, testedIdSet);
		//bestDepthId = findBestDepth_votes(cost, testedIdSet);

		_depthMaps.data[currentPixelIdx] = depth[bestDepthId];
		assignOrientationMap(_orientationMap, currentPixelIdx, orientation[bestDepthId]);

		//8) test the untested sample
		/*std::vector<double> costWithBestDepth; 
		costWithBestDepth.resize( static_cast<int>( _distributionMap.d), UNSET);*/
		for(int j = 0; j<testedIdSet.size(); j++)
		{
			if(testedIdSet[j] == false)	// not tested before
			{					
				computeCost(costWithBestDepth[j], refPixelColor, refPixelPos, refNormColor, deviationColor, j, depth[bestDepthId], numOfPixels, otherImagePixelPos, otherImagePixelColor, orientation[bestDepthId]); // cost is the output
			}
			else
				costWithBestDepth[j] = cost[j*3 + bestDepthId] ;
		}		

		//9) update the distribution
		UpdateDistributionMap(costWithBestDepth, currentPixel, _distributionMap, prob);

}


void patchMatch:: leftToRight()
{
	double ref_h = (_imgStruct_1[0].h);
	double ref_w = (_imgStruct_1[0].w);
	//double ref_d = (_imgStruct_1[0].d);

	//size_t numOfImages = _imgStruct_2.size();
	//for(double row = 300; row < 411; row += 1.0)
	//for(double row = 0; row < ref_h; row += 1.0)
#ifdef USE_OPENMP
	#pragma  omp parallel for schedule(dynamic) 	num_threads(_numOfThreadsUsed)
#endif
	for(int row = 0; row < static_cast<int>(ref_h); row += 1)
	//int row = 300;
	{	
		int maxNumOfPixels = static_cast<int>(pow(_halfWindowSize * 2 + 1, 2));
		pixelPos *refPixelPos = new pixelPos[maxNumOfPixels];
		std::vector<pixelColor> refPixelColor;
		refPixelColor.resize(maxNumOfPixels);
		std::vector<pixelColor> refNormColor;
		refNormColor.resize(maxNumOfPixels);

		pixelPos formerPixel;
		pixelPos currentPixel;
		int formerPixelIdx;
		int currentPixelIdx;
		double depth[3];	// three candidate depth			
		std::vector<int> imageLayerId[2]; 			
		std::vector<double> cost; 
		cost.resize(3 * static_cast<int>(_distributionMap.d), UNSET);	
		double colStart; double colEnd;
		double rowStart; double rowEnd;	
		int numOfPixels;
		std::vector<double> costWithBestDepth; 
		costWithBestDepth.resize( static_cast<int>( _distributionMap.d), UNSET);
		std::vector<bool> testedIdSet;
		testedIdSet.resize(static_cast<int>( _distributionMap.d));
		int bestDepthId;

		std::vector<double> prob; 
		prob.resize(static_cast<int>( _distributionMap.d));
		std::vector<pixelPos> otherImagePixelPos;
		otherImagePixelPos.resize(maxNumOfPixels);
		std::vector<pixelColor> otherImagePixelColor;
		otherImagePixelColor.resize(maxNumOfPixels);

		cv::Mat orientation[3];
		for(int i=0; i<3; i++)
			orientation[i].create(1,3,CV_64F);
		//for(double col = 299; col <= 399; col+=1.0)
		for(double col = 1; col < ref_w; col +=1.0)
		{			
			wrap1(static_cast<double>(row), col, colStart, colEnd, rowStart, rowEnd, numOfPixels, refPixelPos, refPixelColor, 
				ref_w, ref_h);
			//4) draw random depth value, and image id			
			formerPixel._pt.at<double>(0) = col - 1;  formerPixel._pt.at<double>(1) = static_cast<double>(row); 
			currentPixel._pt.at<double>(0) = col;  currentPixel._pt.at<double>(1) = static_cast<double>(row); 
			
			wrap2(formerPixelIdx, currentPixelIdx, depth,
				orientation, imageLayerId, numOfPixels,
				formerPixel, currentPixel, refPixelColor, refNormColor, refPixelPos,
				cost, testedIdSet, bestDepthId, 
				costWithBestDepth, otherImagePixelPos, otherImagePixelColor, prob);						
		}			
		delete []refPixelPos;
	}	
}

void patchMatch::assignOrientationMap(dataMap &orientationMap, const int &currentPixelIdx, const cv::Mat &orientation)
{
	orientationMap.data[currentPixelIdx] = 	orientation.at<double>(0);
	orientationMap.data[currentPixelIdx + static_cast<int>(orientationMap.arraySize)] = orientation.at<double>(1);
	orientationMap.data[currentPixelIdx + static_cast<int>(orientationMap.arraySize) * 2] = orientation.at<double>(2);
}

void patchMatch:: TopToDown()
{
	double ref_h = (_imgStruct_1[0].h);
	double ref_w = (_imgStruct_1[0].w);
	//double ref_d = (_imgStruct_1[0].d);

	//size_t numOfImages = _imgStruct_2.size();
	//for(double row = 300; row < 411; row += 1.0)
	//for(double row = 0; row < ref_h; row += 1.0)
#ifdef USE_OPENMP
	#pragma  omp parallel for schedule(dynamic, 1)  num_threads(_numOfThreadsUsed)	
#endif
	for(int col = 0; col < static_cast<int>(ref_w); col += 1)
	{	
		int maxNumOfPixels = static_cast<int>(pow(_halfWindowSize * 2 + 1, 2));
		pixelPos *refPixelPos = new pixelPos[maxNumOfPixels];
		std::vector<pixelColor> refPixelColor;
		refPixelColor.resize(maxNumOfPixels);
		std::vector<pixelColor> refNormColor;
		refNormColor.resize(maxNumOfPixels);

		pixelPos formerPixel;
		pixelPos currentPixel;
		int formerPixelIdx;
		int currentPixelIdx;
		double depth[3];	// three candidate depth			
		std::vector<int> imageLayerId[2]; 			
		std::vector<double> cost; 
		cost.resize(3 * static_cast<int>(_distributionMap.d), UNSET);	
		double colStart; double colEnd;
		double rowStart; double rowEnd;	
		int numOfPixels;
		std::vector<double> costWithBestDepth; 
		costWithBestDepth.resize( static_cast<int>( _distributionMap.d), UNSET);
		std::vector<bool> testedIdSet;
		testedIdSet.resize(static_cast<int>( _distributionMap.d));
		int bestDepthId;

		std::vector<double> prob; 
		prob.resize(static_cast<int>( _distributionMap.d));
		std::vector<pixelPos> otherImagePixelPos;
		otherImagePixelPos.resize(maxNumOfPixels);
		std::vector<pixelColor> otherImagePixelColor;
		otherImagePixelColor.resize(maxNumOfPixels);

		cv::Mat orientation[3];
		for(int i=0; i<3; i++)
			orientation[i].create(1,3,CV_64F);

		//for(double col = 299; col <= 399; col+=1.0)
		for(double row = 1.0; row < ref_h; row += 1.0)
		{	
			wrap1(row,static_cast<double>(col), colStart, colEnd, rowStart, rowEnd, numOfPixels, refPixelPos, refPixelColor, 
				ref_w, ref_h);
			//4) draw random depth value, and image id
			//pixelPos formerPixel(col-1, row);   // ***
			//pixelPos currentPixel(col, row);
			formerPixel._pt.at<double>(0) = col;  formerPixel._pt.at<double>(1) = static_cast<double>(row - 1); 
			currentPixel._pt.at<double>(0) = col;  currentPixel._pt.at<double>(1) = static_cast<double>(row); 
			
			wrap2(formerPixelIdx, currentPixelIdx, depth,
				orientation, imageLayerId, numOfPixels,
				formerPixel, currentPixel, refPixelColor, refNormColor, refPixelPos,
				cost, testedIdSet, bestDepthId, 
				costWithBestDepth, otherImagePixelPos, otherImagePixelColor, prob);			
			
		}			
		delete []refPixelPos;
	}	
}

void patchMatch:: DownToTop()
{
	double ref_h = (_imgStruct_1[0].h);
	double ref_w = (_imgStruct_1[0].w);
	//double ref_d = (_imgStruct_1[0].d);

	//size_t numOfImages = _imgStruct_2.size();
	//for(double row = 300; row < 411; row += 1.0)
	//for(double row = 0; row < ref_h; row += 1.0)
#ifdef USE_OPENMP
	#pragma  omp parallel for schedule(dynamic, 1) 	num_threads(_numOfThreadsUsed)
#endif
	for(int col = 0; col < static_cast<int>(ref_w); col += 1)
	{	
		int maxNumOfPixels = static_cast<int>(pow(_halfWindowSize * 2 + 1, 2));
		pixelPos *refPixelPos = new pixelPos[maxNumOfPixels];
		std::vector<pixelColor> refPixelColor;
		refPixelColor.resize(maxNumOfPixels);
		std::vector<pixelColor> refNormColor;
		refNormColor.resize(maxNumOfPixels);

		pixelPos formerPixel;
		pixelPos currentPixel;
		int formerPixelIdx;
		int currentPixelIdx;
		double depth[3];	// three candidate depth			
		std::vector<int> imageLayerId[2]; 			
		std::vector<double> cost; 
		cost.resize(3 * static_cast<int>(_distributionMap.d), UNSET);	
		double colStart; double colEnd;
		double rowStart; double rowEnd;	
		int numOfPixels;
		std::vector<double> costWithBestDepth; 
		costWithBestDepth.resize( static_cast<int>( _distributionMap.d), UNSET);
		std::vector<bool> testedIdSet;
		testedIdSet.resize(static_cast<int>( _distributionMap.d));
		int bestDepthId;

		std::vector<double> prob; 
		prob.resize(static_cast<int>( _distributionMap.d));
		std::vector<pixelPos> otherImagePixelPos;
		otherImagePixelPos.resize(maxNumOfPixels);
		std::vector<pixelColor> otherImagePixelColor;
		otherImagePixelColor.resize(maxNumOfPixels);

		cv::Mat orientation[3];
		for(int i=0; i<3; i++)
			orientation[i].create(1,3,CV_64F);

		//for(double col = 299; col <= 399; col+=1.0)
		for(double row = ref_h - 2; row >= 0; row -= 1.0)
		{			
			////1) find the start and end of the row and col (start smaller than end)
			//findRange( static_cast<double>(row), col, rowStart, rowEnd, colStart, colEnd, _halfWindowSize, ref_w, ref_h);	

			////2)						
			//numOfPixels = static_cast<int>(((colEnd - colStart + 1) * (rowEnd - rowStart + 1))); 
			//findPixelPos(refPixelPos, numOfPixels ,rowStart, rowEnd, colStart, colEnd);	// 
			//
			////3) find the color in reference image given pixels			
			////std::vector<pixelColor> refPixelColor;
			////refPixelColor.reserve(numOfPixels);
			//findPixelColors(refPixelColor, refPixelPos, _imgStruct_1[0], numOfPixels);	// within this function, it should allow non-integer pixel positions

			wrap1(row,static_cast<double>(col), colStart, colEnd, rowStart, rowEnd, numOfPixels, refPixelPos, refPixelColor, 
				ref_w, ref_h);

			//4) draw random depth value, and image id
			//pixelPos formerPixel(col-1, row);   // ***
			//pixelPos currentPixel(col, row);
			formerPixel._pt.at<double>(0) = col;  formerPixel._pt.at<double>(1) = static_cast<double>(row + 1); 
			currentPixel._pt.at<double>(0) = col;  currentPixel._pt.at<double>(1) = static_cast<double>(row); 
			
			wrap2(formerPixelIdx, currentPixelIdx, depth,
				orientation, imageLayerId, numOfPixels,
				formerPixel, currentPixel, refPixelColor, refNormColor, refPixelPos,
				cost, testedIdSet, bestDepthId, 
				costWithBestDepth, otherImagePixelPos, otherImagePixelColor, prob);			
		}			
		delete []refPixelPos;
	}	
}


void patchMatch:: RightToLeft()
{
	double ref_h = (_imgStruct_1[0].h);
	double ref_w = (_imgStruct_1[0].w);
	//double ref_d = (_imgStruct_1[0].d);

	//size_t numOfImages = _imgStruct_2.size();

	//for(double row = 300; row < 411; row += 1.0)

	//for(double row = 0; row < ref_h; row += 1.0)
#ifdef USE_OPENMP
	#pragma  omp parallel for schedule(dynamic, 1) 	num_threads(_numOfThreadsUsed)
#endif
	for(int row = 0; row < static_cast<int>(ref_h); row += 1)
	{	
		int maxNumOfPixels = static_cast<int>(pow(_halfWindowSize * 2 + 1, 2));
		pixelPos *refPixelPos = new pixelPos[maxNumOfPixels];
		std::vector<pixelColor> refPixelColor;
		refPixelColor.resize(maxNumOfPixels);
		std::vector<pixelColor> refNormColor;
		refNormColor.resize(maxNumOfPixels);

		pixelPos formerPixel;
		pixelPos currentPixel;
		int formerPixelIdx;
		int currentPixelIdx;
		double depth[3];	// three candidate depth			
		std::vector<int> imageLayerId[2]; 			
		std::vector<double> cost; 
		cost.resize(3 * static_cast<int>(_distributionMap.d), UNSET);	
		double colStart; double colEnd;
		double rowStart; double rowEnd;	
		int numOfPixels;
		std::vector<double> costWithBestDepth; 
		costWithBestDepth.resize( static_cast<int>( _distributionMap.d), UNSET);
		std::vector<bool> testedIdSet;
		testedIdSet.resize(static_cast<int>( _distributionMap.d));
		int bestDepthId;

		std::vector<double> prob; 
		prob.resize(static_cast<int>( _distributionMap.d));
		std::vector<pixelPos> otherImagePixelPos;
		otherImagePixelPos.resize(maxNumOfPixels);
		std::vector<pixelColor> otherImagePixelColor;
		otherImagePixelColor.resize(maxNumOfPixels);

		cv::Mat orientation[3];
		for(int i=0; i<3; i++)
			orientation[i].create(1,3,CV_64F);

		//for(double col = 299; col <= 399; col+=1.0)
		for(double col = ref_w - 2; col >= 0; col -=1.0)
		{			
			
			wrap1(static_cast<double>(row) ,col, colStart, colEnd, rowStart, rowEnd, numOfPixels, refPixelPos, refPixelColor, 
				ref_w, ref_h);

			//4) draw random depth value, and image id
			//pixelPos formerPixel(col-1, row);   // ***
			//pixelPos currentPixel(col, row);
			formerPixel._pt.at<double>(0) = col + 1;  formerPixel._pt.at<double>(1) = static_cast<double>(row); 
			currentPixel._pt.at<double>(0) = col;  currentPixel._pt.at<double>(1) = static_cast<double>(row); 
			
			wrap2(formerPixelIdx, currentPixelIdx, depth,
				orientation, imageLayerId, numOfPixels,
				formerPixel, currentPixel, refPixelColor, refNormColor, refPixelPos,
				cost, testedIdSet, bestDepthId, 
				costWithBestDepth, otherImagePixelPos, otherImagePixelColor, prob);			
		}			
		delete []refPixelPos;
	}	
}


pixelColor patchMatch::meanNormalize(const std::vector<pixelColor> &refPixelColor, std::vector<pixelColor> &refNormColor, const int &numOfPixels)
{	

	pixelColor meanColor(0.0,0.0,0.0);
	for(int i = 0; i<numOfPixels; i++)
	{
		meanColor._color += refPixelColor[i]._color;
	}
	meanColor._color /= numOfPixels;

	pixelColor deviationColor(0.0, 0.0, 0.0);
	for(int i = 0; i<numOfPixels; i++)
	{
		refNormColor[i]._color = refPixelColor[i]._color - meanColor._color;
		deviationColor._color += (refNormColor[i]._color.mul(refNormColor[i]._color));		
	}	
	deviationColor.sqrtRoot();

	/*for(int i = 0; i<numOfPixels; i++)
		for(int j = 0; j<3; j++)
		{
			if(deviationColor._color.at<double>(j) != 0)
				refNormColor[i]._color.at<double>(j) /= deviationColor._color.at<double>(j);
		}*/
	return deviationColor;
	
}

void patchMatch:: UpdateDistributionMap(const std::vector<double> &cost, const pixelPos &currentPos, const dataMap & distributionMap, std::vector<double> &prob)
{
	//std::vector<double> prob;
	//prob.resize(cost.size());
	double variance_inv = 1.0/(_sigma * _sigma);
	for(int i = 0; i<cost.size(); i++)
	{
		prob[i] = exp(-0.5 * (1-cost[i]) * (1-cost[i]) * variance_inv);
	}
	// do not do normalization

	/* double sum = 0;
	for(int i =0; i<prob.size(); i++)
	{
		sum += prob[i];
	}
	for(int i = 0; i< cost.size(); i++)
	{
		distributionMap.data[static_cast<int>(currentPos.sub2Idx(distributionMap.h, distributionMap.arraySize, i))] = prob[i]/sum;
	}	*/

	for(int i = 0; i<cost.size(); i++)
	{
		distributionMap.data[static_cast<int>(currentPos.sub2Idx(distributionMap.h, distributionMap.arraySize, i))] = prob[i];
	}
}

int patchMatch:: findBestDepth_votes(const std::vector<double> &cost, std::vector<bool> &testedIdSet)
{
	int depthVotes[3] = {0}; 
	for(int i = 0; i < _distributionMap.d; i++)
	{
		if(cost[i * 3] != UNSET)
		{
			double maxCost = std::max(std::max(cost[i*3], cost[i*3 + 1]), cost[i*3 + 2]);
			if(cost[i * 3] == maxCost)
				++depthVotes[0];
			if(cost[i * 3 + 1] == maxCost)
				++depthVotes[1];
			if(cost[i * 3 + 2] == maxCost)
				++depthVotes[2];			
		}		
	}
	//int maxmumVotes = max(depthVotes[0], max(depthVotes[1], depthVotes[2]));
	bool repeated = false;
	int maxVotes = depthVotes[0];
	int bestDepthId = 0;
	for(int i = 1; i<3; i++)
	{
		if(depthVotes[i] == maxVotes)
			repeated = true;
		if(depthVotes[i] > maxVotes)
		{
			maxVotes = depthVotes[i];
			bestDepthId = i;
			repeated = false;			
		}
	}
	if(!repeated)
		return bestDepthId;
	else
		return findBestDepth_average(cost, testedIdSet);
}

int patchMatch::findBestDepth_average(const std::vector<double> &cost, std::vector<bool> &testedIdSet)
{

	double averageCost[3] = {0}; //double numOfImagesTested = 0.;
	double num[3] = {0};
	//std::vector<bool> unsetLists;
	for(int i = 0; i < _distributionMap.d; i++)
	{
		if(cost[i * 3] != UNSET)
		{
			
			for(int j = 0; j<3; j++)
			{
				//if(cost[i*3 + j] != -1)
				//{
					averageCost[j] += cost[i*3 + j];		
					//++num[j];
				//}
			}			
			//numOfImagesTested++;
			testedIdSet[i] = true;
		}
		else
		{
			testedIdSet[i] = false;	// not tested
		}
	}

	/*for(int i = 0; i<3; i++)	
	{
		if(num[i] != 0)
			averageCost[i] /= (num[i]);
		else
			averageCost[i] = -1.0;
	}*/

	double maxCost = averageCost[0];
	int bestDepthId = 0;
	for(int i = 1; i<3; i++)
	{
		if(averageCost[i] > maxCost)
		{
			maxCost = averageCost[i];
			bestDepthId = i;
		}
	}
	return bestDepthId;
}
