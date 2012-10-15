#include "parseParms.h"


void patchMatch:: parseImageStructure(std::vector<ImageStruct> *allImgStruct, const mxArray* prhs)
{
	int nfields = mxGetNumberOfFields(prhs);	
	size_t NStructElems = mxGetNumberOfElements(prhs);	
	for(int i = 0; i< NStructElems; i++)
	{
		std::vector<std::string> fnames;
		mxArray *tmp;
		for (int ifield=0; ifield< nfields; ifield++){
			fnames.push_back(std::string(mxGetFieldNameByNumber(prhs,ifield)));	
			tmp = mxGetFieldByNumber(prhs, i, ifield);
			ImageStruct imgStruct;

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
			allImgStruct->push_back(imgStruct);
		}
	}
}

void patchMatch::parseDataMap(dataMap *dataMaps, const mxArray *p)
{
	dataMaps->p = mxDuplicateArray(p);
	dataMaps->data = mxGetPr(dataMaps->p);
	const mwSize *sz = mxGetDimensions(dataMaps->p);
	dataMaps->h = static_cast<double>(sz[0]);
	dataMaps->w = static_cast<double>(sz[1]);
	dataMaps->d = static_cast<double>(sz[2]);
	
}

void patchMatch::findRange(const int &row, const int &col, int &rowStart, int &rowEnd, int &colStart, int &colEnd, const int &halfWindowSize, const int &w, const int &h)
{
	rowStart = row - halfWindowSize > 0 ? (row - halfWindowSize) : 0;
	rowEnd = row + halfWindowSize < h ? (row + halfWindowSize) : h;
	colStart = col - halfWindowSize > 0 ? (col - halfWindowSize) : 0;
	colEnd = col + halfWindowSize < w ? (col + halfWindowSize) : w;
}

void patchMatch::findPixelPos(std::vector<pixelPos> &pixelPostions , const int &rowStart, const int &rowEnd, const int &colStart, const int &colEnd)
{
	for(int i = rowStart; i<rowEnd; i++)
	{
		for(int j = colStart; j<colEnd; j++)
		{
			pixelPostions.push_back(pixelPos(j, i));
		}
	}
}

void patchMatch:: leftToRight()
{
	int ref_h = static_cast<int>(_imgStruct_1[0].h);
	int ref_w = static_cast<int>(_imgStruct_1[0].w);
	int ref_d = static_cast<int>(_imgStruct_1[0].d);

	for(int row = 0; row < ref_h; row++)
	{
		for(int col = 1; col < ref_w; col++)
		{
			// prepare color of current pixels
			//1) find the start and end of the row and col (start smaller than end)
			int colStart; int colEnd;
			int rowStart; int rowEnd;
			findRange(row, col, rowStart, rowEnd, colStart, colEnd, _halfWindowSize, ref_w, ref_h);
			//2) find all the pixels in the current image
			int numOfPixels = (colEnd - colStart + 1) * (rowEnd - rowStart + 1); 
			std::vector<pixelPos> refPixelPos;
			refPixelPos.reserve(numOfPixels);
			findPixelPos(refPixelPos, rowStart, rowEnd, colStart, colEnd);

			//3) find the color given pixels


			//4) transforming the pixels to the other image (depth given)

			//5) find the color given pixels. 

			//6) compare the pixels and calculate the cost.

			

		}
	}
}