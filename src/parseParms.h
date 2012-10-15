#ifndef PARSEPARMS_H
#define PARSEPARMS_H

#include <mex.h>
#include <string>
#include <vector>
//#include <opencv/cxcore.h>

class pixelPos 
{
public:
	double _x;
	double _y;
	pixelPos(double x, double y){_x = x; _y = y;}
};

typedef struct
{
	double *imageData;
	double h;
	double w;
	double d;
	//----------------------------
	std::string imageName;
	double *K;
	double *R;
	double *T;
	double *C;
} ImageStruct;

typedef struct 
{
	mxArray *p;
	double *data;
	double h;
	double w;
	double d;
}dataMap ;

typedef struct  
{
	double r;
	double g;
	double b;
}pixelColor;


class patchMatch{
public:	
	//	
	dataMap _depthMaps;
	dataMap _distributionMap;
	std::vector<ImageStruct> _imgStruct_1;
	std::vector<ImageStruct> _imgStruct_2;
	int _halfWindowSize;

	patchMatch(const mxArray *prhs[])
	{
		parseImageStructure(&_imgStruct_1, prhs[0]);
		parseImageStructure(&_imgStruct_2, prhs[1]);
		parseDataMap(&_depthMaps, prhs[2]);
		parseDataMap(&_distributionMap, prhs[3]);

		_halfWindowSize = 4;
	}
	void parseImageStructure(std::vector<ImageStruct> *allImgStruct, const mxArray* p);
	void parseDataMap(dataMap *dataMaps, const mxArray *p);
	inline pixelColor fetchColorPixel(const double *imageData, const int &ind_r, const int &ind_g, const int &ind_b )
	{
		pixelColor p;
		p.r = imageData[ind_r];
		p.g = imageData[ind_g];
		p.b = imageData[ind_b];
		return p;
	}

	void leftToRight();
	void findRange(const int &row, const int &col, int &rowStart, int &rowEnd, int &colStart, int &colEnd, const int &halfWindowSize, const int &w, const int &h);
	void findPixelPos(std::vector<pixelPos> &pixelPos , const int &rowStart, const int &rowEnd, const int &colStart, const int &colEnd);

};



#endif