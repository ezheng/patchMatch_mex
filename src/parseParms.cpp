#include "parseParms.h"
#include <assert.h>

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
	dataMaps->p = mxDuplicateArray(p);
	dataMaps->data = mxGetPr(dataMaps->p);
	mwSize numOfDimensions = mxGetNumberOfDimensions(p);

	const mwSize *sz = mxGetDimensions(dataMaps->p);
	dataMaps->h = static_cast<double>(sz[0]);
	dataMaps->w = static_cast<double>(sz[1]);
	if(numOfDimensions == 3)
		dataMaps->d = static_cast<double>(sz[2]);
	else
		dataMaps->d = 1;
	
}

void patchMatch::findRange(const double &row, const double &col, double &rowStart, double &rowEnd, double &colStart, double &colEnd, const double &halfWindowSize, const double &w, const double &h)
{
	rowStart = row - halfWindowSize >= 0 ? (row - halfWindowSize) : 0;
	rowEnd = row + halfWindowSize <= h-1 ? (row + halfWindowSize) : h-1;
	colStart = col - halfWindowSize >= 0 ? (col - halfWindowSize) : 0;
	colEnd = col + halfWindowSize <= w-1 ? (col + halfWindowSize) : w-1;
}

void patchMatch::findPixelPos(std::vector<pixelPos> &pixelPostions , const double &rowStart, const double &rowEnd, const double &colStart, const double &colEnd)
{
	for(double i = rowStart; i<=rowEnd; i++)
	{
		for(double j = colStart; j<=colEnd; j++)
		{
			pixelPostions.push_back(pixelPos(j, i));			
		}
	}
}

void patchMatch::findPixelColors(std::vector<pixelColor> &pColors, const std::vector<pixelPos> &pPos, ImageStruct &img)
{
	size_t numOfPixels = pPos.size();
	for(int i = 0; i < numOfPixels; i++)
	{
		double ind_r, ind_g, ind_b;
		pPos[i].sub2Idx(ind_r, ind_g, ind_b, img.h, img.w);
		pColors.push_back( fetchColorOnePixel(img.imageData, static_cast<int>(ind_r), static_cast<int>(ind_g), static_cast<int>(ind_b)));
	}

}

void patchMatch::findPixelColorsInterpolation(std::vector<pixelColor> &pColors, const std::vector<pixelPos> &pPos, ImageStruct &img)
{
	size_t numOfPixels = pPos.size();
	std::vector<double> ind_r;
	std::vector<double> ind_g;
	std::vector<double> ind_b;
	std::vector<double> weight;

	for(int i = 0; i < numOfPixels; i++)
	{
		pixelColor pCol(0.0, 0.0, 0.0);
		
		if( pPos[i]._pt.at<double>(0) < 0.5 || pPos[i]._pt.at<double>(0) > pPos[i]._pt.cols-0.5 ||
			pPos[i]._pt.at<double>(1) < 0.5 || pPos[i]._pt.at<double>(0) > pPos[i]._pt.rows-0.5 )
		{
			pCol._color.at<double>(3) = 1.0;			
		}
		else
		{
			pPos[i].sub2Idx(weight, ind_r, ind_g, ind_b, img.h, img.w);				
			for(int j = 0; j<ind_r.size(); j++)
			{
				pCol += fetchColorOnePixel(img.imageData, static_cast<int>(ind_r[j]), static_cast<int>(ind_g[j]), static_cast<int>(ind_b[j])) *  weight[j] ;	// do interpolation
			}
		}		
		pColors.push_back(pCol);
	}
}


pixelColor patchMatch::fetchColorOnePixel(const double *imageData, const int &ind_r, const int &ind_g, const int &ind_b )
{
	return pixelColor(imageData[ind_r], imageData[ind_g], imageData[ind_b]);		
	
}

void patchMatch::updateDistribution(const pixelPos &formerPixel, const pixelPos &currentPixel, dataMap& distributionMap)
{	
	// update distribution for each layer
	for(int layer = 0; layer < distributionMap.d; layer++)
	{
		double curIdx = currentPixel.sub2Idx(distributionMap.h, distributionMap.w, layer );
		double formerIdx = formerPixel.sub2Idx(distributionMap.h, distributionMap.w, layer);
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
	// draw samples based on
		
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
		distribution[i] = _distributionMap.data[static_cast<int>( curPixel.sub2Idx(distributionMap.h, distributionMap.w, i))];
	}
	// normalize distribution:
	// normalizeDistribution(distribution);
	drawSamples(distribution, numOfSamples, imageLayerId);
}

void patchMatch:: computeCost(double &cost, const std::vector<pixelColor> &refPixelColor, 
	const std::vector<pixelPos> &refPixelPos, int imageId, double depth)
{
	int numOfPixels = static_cast<int>(refPixelPos.size());	
	std::vector<pixelPos> otherImagePixelPos;	
	getOtherImagePixelPos(otherImagePixelPos, refPixelPos, depth, imageId);
	std::vector<pixelColor> otherImagePixelColor;
	findPixelColorsInterpolation(otherImagePixelColor, otherImagePixelPos, _imgStruct_2[imageId]);	// 
	
	// calculate the cost using 1) refPixelColor, and 2) 
	
	cost = calculateNCC( otherImagePixelColor, refPixelColor);
}

void patchMatch::getOtherImagePixelPos(std::vector<pixelPos> &otherImagePixelPos, const std::vector<pixelPos> &refPixelPos, double depth, int imageId)
{
	cv::Mat normalVector = (cv::Mat_<double>(1,3) << 0, 0, 1);

	cv::Mat opencv_R = _imgStruct_2[imageId].opencv_R *_imgStruct_1[0].opencv_inverseR ;
	cv::Mat opencv_T = _imgStruct_2[imageId].opencv_R * (_imgStruct_2[imageId].opencv_C - _imgStruct_1[0].opencv_C);	
	cv::Mat H = _imgStruct_2[imageId].opencv_K * (opencv_R - opencv_T * normalVector / depth) * _imgStruct_1[0].opencv_inverseK;

	int numOfPixels = static_cast<int>(refPixelPos.size());
	otherImagePixelPos.reserve(numOfPixels);
	for(int i = 0; i<numOfPixels; i++)
	{
		pixelPos refImageOnePixel(refPixelPos[i]._pt.at<double>(0) + 0.5, refPixelPos[i]._pt.at<double>(1) + 0.5 );
		cv::Mat newPixel = H * refImageOnePixel._pt;
		// normalize:
		newPixel = newPixel/newPixel.at<double>(2);

		otherImagePixelPos.push_back(pixelPos(newPixel)); 
	}

}


double patchMatch::calculateNCC(const std::vector<pixelColor> &otherImagePixelColor, const std::vector<pixelColor> &refPixelColor)
{
	assert(otherImagePixelColor.size() == refPixelColor.size());
	int numOfPixels = static_cast<int>(otherImagePixelColor.size());
	
	pixelColor otherImageMean(0., 0., 0.);
	pixelColor refImageMean(0.,0.,0.);	
	int numOfValidPixels = 0;
	for(int i = 0; i < numOfPixels; i++)
	{
		if(otherImagePixelColor[i]._color.at<double>(3) != 1.0f)
		{
			otherImageMean += otherImagePixelColor[i];
			refImageMean += refPixelColor[i];
			numOfValidPixels += 1;			
		}		
	}
	if(numOfValidPixels == 0)
	{
		return -1;	
	}
	otherImageMean = otherImageMean * (1/ static_cast<double>(numOfValidPixels));
	refImageMean = refImageMean * (1/ static_cast<double>(numOfPixels));
	// 
	std::vector<pixelColor> otherImageSubtracted;
	std::vector<pixelColor> refImageSubtracted;
	for(int i= 0; i < numOfPixels; i++ )
	{
		if(otherImagePixelColor[i]._color.at<double>(3) != 1.0f)
		{
			otherImageSubtracted.push_back( otherImagePixelColor[i] - otherImageMean );
			refImageSubtracted.push_back( refPixelColor[i] - refImageMean ); 
		}
	}
	//
	pixelColor otherImageSigma;
	pixelColor refImageSigma;
	for(int i = 0; i< numOfValidPixels; i++)
	{
		otherImageSigma += (otherImageSubtracted[i] * otherImageSubtracted[i]);
		refImageSigma += (refImageSubtracted[i] * refImageSubtracted[i]);
	}
	otherImageSigma = otherImageSigma * (1/ static_cast<double>(numOfValidPixels));
	refImageSigma = refImageSigma * (1/ static_cast<double>(numOfValidPixels));

	double cost_rgb[3];
	for(int i = 0; i<3; i++)
	{
		for(int j = 0; j< numOfValidPixels; j++)
		{
			cost_rgb[i] = otherImageSubtracted[j]._color.at<double>(i) * refImageSubtracted[j]._color.at<double>(i) / otherImageSigma._color.at<double>(i) / refImageSigma._color.at<double>(i);
		}
	}
	double cost = (cost_rgb[0] + cost_rgb[1] + cost_rgb[2])/3.0f/static_cast<double>(numOfValidPixels);
	return cost;
}

void patchMatch:: leftToRight()
{
	double ref_h = (_imgStruct_1[0].h);
	double ref_w = (_imgStruct_1[0].w);
	double ref_d = (_imgStruct_1[0].d);

	size_t numOfImages = _imgStruct_2.size();

	for(double row = 0; row < ref_h; row+=1.0)
	{
		mexPrintf("%d is started\n", row);
		for(double col = 1; col < ref_w; col+=1.0)
		{			
			//1) find the start and end of the row and col (start smaller than end)
			double colStart; double colEnd;
			double rowStart; double rowEnd;
			findRange(row, col, rowStart, rowEnd, colStart, colEnd, _halfWindowSize, ref_w, ref_h);	// though these values are double type, but it is integer

			//2) find all the pixels in the current image
			int numOfPixels = static_cast<int>(((colEnd - colStart + 1) * (rowEnd - rowStart + 1))); 
			std::vector<pixelPos> refPixelPos;
			refPixelPos.reserve(numOfPixels);
			findPixelPos(refPixelPos, rowStart, rowEnd, colStart, colEnd);	// 

			//3) find the color in reference image given pixels
			std::vector<pixelColor> refPixelColor;
			refPixelColor.reserve(numOfPixels);
			findPixelColors(refPixelColor, refPixelPos, _imgStruct_1[0]);	// within this function, it should allow non-integer pixel positions

			//4) draw random depth value, and image id
			pixelPos formerPixel(col-1, row);   // ***
			pixelPos currentPixel(col, row);
			int formerPixelIdx = static_cast<int>(formerPixel.sub2Idx(_depthMaps.h));
			int currentPixelIdx = static_cast<int>(currentPixel.sub2Idx(_depthMaps.h));
			double depth[3];	// three candidate depth			
			depth[0] = _depthMaps.data[formerPixelIdx];
			depth[1] = _depthRandomMaps.data[currentPixelIdx];
			depth[2] = _depthMaps.data[currentPixelIdx];			
			//5) draw samples and update the image ID distribution
			// draw samples:
			std::vector<int> imageLayerId[2]; 
			drawSamples(_distributionMap, imageLayerId[0], _numOfSamples, formerPixel);
			// update:
			//updateDistribution(formerPixel, currentPixel, _distributionMap);
			// draw samples				
			drawSamples(_distributionMap, imageLayerId[1], _numOfSamples, currentPixel);

			//6) transforming the pixels to the other image (depth given, image id is given) and find the color for given pixels. calculate costs
			std::vector<double> cost; 
			cost.resize(3 * static_cast<int>(_distributionMap.d), UNSET);			
			for(int i = 0; i< 3; i++)
			{
				for(int j = 0; j <imageLayerId->size(); j++ )
					for(int k = 0; k < imageLayerId[j].size(); k++ )
					{
						if(cost[i + 3 * imageLayerId[j][k]] == UNSET)
						{
							computeCost(cost[i + 3 * imageLayerId[j][k]], refPixelColor, refPixelPos, imageLayerId[j][k], depth[i]); // cost is the output
						}
					}
			}
			//7) based on the cost, VOTE which depth to use. and then save the depth
			std::vector<bool> testedIdSet;
			int bestDepthId = findBestDepth_average(cost, testedIdSet);
			_depthMaps.data[currentPixelIdx] = depth[bestDepthId];
			
			//8) test the untested sample
			std::vector<double> costWithBestDepth; costWithBestDepth.resize( static_cast<int>( _distributionMap.d), UNSET);
			for(int j = 0; j<testedIdSet.size(); j++)
			{
				if(testedIdSet[j] == false)
				{					
					computeCost(costWithBestDepth[j], refPixelColor, refPixelPos, j, depth[bestDepthId]); // cost is the output
				}
				else
					costWithBestDepth[j] = cost[j*3 + bestDepthId] ;
			}		

			//9) update the distribution
			UpdateDistributionMap(costWithBestDepth, currentPixel, _distributionMap);
		}
	}
}

void patchMatch:: UpdateDistributionMap(const std::vector<double> &cost, const pixelPos &currentPos, const dataMap & distributionMap)
{
	std::vector<double> prob;
	prob.resize(cost.size());
	for(int i = 0; i<cost.size(); i++)
	{
		prob[i] = exp(-0.5 * (1-cost[i]) * (1-cost[i])/(_sigma * _sigma));
	}
	double sum = 0;
	for(int i =0; i<prob.size(); i++)
	{
		sum += prob[i];
	}
	for(int i = 0; i< cost.size(); i++)
	{
		distributionMap.data[static_cast<int>(currentPos.sub2Idx(distributionMap.h, distributionMap.w, i))] = prob[i]/sum;
	}	
}

int patchMatch::findBestDepth_average(const std::vector<double> &cost, std::vector<bool> &testedIdSet)
{

	double averageCost[3] = {0}; double numOfImagesTested = 0.;
	//std::vector<bool> unsetLists;
	for(int i = 0; i < _distributionMap.d; i++)
	{
		if(cost[i * 3] != UNSET)
		{
			for(int j = 0; j<3; j++)
				averageCost[j] += cost[i*3 + j];			
			numOfImagesTested++;
			testedIdSet.push_back(true);
		}
		else
		{
			testedIdSet.push_back(false);
		}
	}
	for(int i = 0; i<3; i++)
	{
		averageCost[i] /= numOfImagesTested;
	}

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
