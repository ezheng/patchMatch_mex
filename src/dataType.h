#ifndef DATATYPE_H
#define DATATYPE_H

#include <opencv/cxcore.h>
#include <iostream>

class pixelPos 
{
public:
	cv::Mat _pt;		
	pixelPos(double x, double y)
	{ 		
		_pt = (cv::Mat_<double>(3,1)<<x, y, 1.0);
	}
	pixelPos(const pixelPos &p)
	{
		this->_pt = p._pt;
	}
	pixelPos()
	{
		//_pt.create(3,1,CV_64F);
		_pt = (cv::Mat_<double>(3,1)<<0, 0, 1.0);
	}

	pixelPos(cv::Mat pt)
	{
		_pt = pt;
	}

	//inline void sub2Idx(double &ind_r, double &ind_g, double &ind_b, const double &height, const double &width) const
	inline void sub2Idx(double &ind_r, double &ind_g, double &ind_b, const double &height, const double &arraySize) const
	{
		ind_r = _pt.at<double>(0) * height + _pt.at<double>(1);
		ind_g = ind_r +  arraySize;
		ind_b = ind_g +  arraySize;
	}
	inline double sub2Idx(const double &height) const
	{
		return _pt.at<double>(0) * height + _pt.at<double>(1);
	}
	inline double sub2Idx(const double &height, const double &arraySize, int level) const
	{
		return arraySize * level + _pt.at<double>(0) * height + _pt.at<double>(1);
	}

	inline void sub2Idx(double *weight, double *ind_r, const double &height) const
	{
		double x_low = floor( _pt.at<double>(0));
		//double x_heigh = ceil( _pt.at<double>(0));
		double x_heigh = x_low + 1;
		double y_low = floor( _pt.at<double>(1));
		double y_heigh = y_low + 1;
		//double y_heigh = ceil( _pt.at<double>(1));

		//weight.resize(4);
		weight[0] = (x_heigh - _pt.at<double>(0)) * (y_heigh - _pt.at<double>(1));
		weight[1] = (x_heigh - _pt.at<double>(0)) * ( _pt.at<double>(1) - y_low );
		weight[2] = (_pt.at<double>(0) - x_low)    * (y_heigh - _pt.at<double>(1));		
		weight[3] = (_pt.at<double>(0) - x_low)    * ( _pt.at<double>(1) - y_low );

		//ind_r.resize(4); ind_g.resize(4); ind_b.resize(4);
		//double ind_r_int, ind_g_int, ind_b_int;
		//pixelPos pt(x_low, y_low);
		//pt._pt.at<double>(0) = x_low;	pt._pt.at<double>(1) = y_low;
		//pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, arraySize);
		//ind_r[0] = ind_r_int;
		//ind_g[0] = ind_g_int;
		//ind_b[0] = ind_b_int;
		pixelPos pt(x_low, y_low);
		ind_r[0] = pt.sub2Idx(height);
		
		//pt._pt.at<double>(0) = x_low;	pt._pt.at<double>(1) = y_heigh;
		//pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, arraySize);
		ind_r[1] = ind_r[0] + 1;		

		//pt._pt.at<double>(0) = x_heigh;	pt._pt.at<double>(1) = y_low;
		//pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, arraySize);
		//ind_r[2] = ind_r_int + height;
		//ind_g[2] = ind_g_int + height;
		//ind_b[2] = ind_b_int + height;
		ind_r[2] = ind_r[0] + height;
		
		//pt._pt.at<double>(0) = x_heigh;	pt._pt.at<double>(1) = y_heigh;
		//pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, arraySize);
		//ind_r[3] = ind_r_int + 1;
		//ind_g[3] = ind_g_int + 1;
		//ind_b[3] = ind_b_int + 1;
		ind_r[3] = ind_r[2] + 1;
		
	}

	inline void sub2Idx(double *weight, double *ind_r, double *ind_g, double *ind_b, const double &height, const double &arraySize) const
	{
		double x_low = floor( _pt.at<double>(0));
		//double x_heigh = ceil( _pt.at<double>(0));
		double x_heigh = x_low + 1;
		double y_low = floor( _pt.at<double>(1));
		double y_heigh = y_low + 1;
		//double y_heigh = ceil( _pt.at<double>(1));
				
		weight[0] = (x_heigh - _pt.at<double>(0)) * (y_heigh - _pt.at<double>(1));
		weight[1] = (x_heigh - _pt.at<double>(0)) * ( _pt.at<double>(1) - y_low );
		weight[2] = (_pt.at<double>(0) - x_low)    * (y_heigh - _pt.at<double>(1));		
		weight[3] = (_pt.at<double>(0) - x_low)    * ( _pt.at<double>(1) - y_low );
		//weight[3] = 1 - weight[0] - weight[1] - weight[2];
				
		double ind_r_int, ind_g_int, ind_b_int;
		pixelPos pt(x_low, y_low);
		//pt._pt.at<double>(0) = x_low;	pt._pt.at<double>(1) = y_low;
		pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, arraySize);
		ind_r[0] = ind_r_int;
		ind_g[0] = ind_g_int;
		ind_b[0] = ind_b_int;
		
		//pt._pt.at<double>(0) = x_low;	pt._pt.at<double>(1) = y_heigh;
		//pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, arraySize);
		ind_r[1] = ind_r[0] + 1;
		ind_g[1] = ind_g[0]  + 1;
		ind_b[1] = ind_b[0] + 1;

		//pt._pt.at<double>(0) = x_heigh;	pt._pt.at<double>(1) = y_low;
		//pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, arraySize);
		//ind_r[2] = ind_r_int + height;
		//ind_g[2] = ind_g_int + height;
		//ind_b[2] = ind_b_int + height;
		ind_r[2] = ind_r[0] + height;
		ind_g[2] = ind_g[0] + height;
		ind_b[2] = ind_b[0] + height;

		//pt._pt.at<double>(0) = x_heigh;	pt._pt.at<double>(1) = y_heigh;
		//pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, arraySize);
		//ind_r[3] = ind_r_int + 1;
		//ind_g[3] = ind_g_int + 1;
		//ind_b[3] = ind_b_int + 1;
		ind_r[3] = ind_r[2] + 1;
		ind_g[3] = ind_g[2] + 1;
		ind_b[3] = ind_b[2] + 1;
	}

};


class ImageStruct
{
public:
	double *imageData;
	double h;
	double w;
	double d;
	double arraySize;
	//----------------------------
	std::string imageName;
	double *K;
	double *R;
	double *T;
	double *C;

	// 
	cv::Mat opencv_K;
	cv::Mat opencv_R;
	cv::Mat opencv_T;
	cv::Mat opencv_C;

	cv::Mat opencv_inverseK;
	cv::Mat opencv_inverseR;

	cv::Mat opencv_relative_R;
	cv::Mat opencv_relative_T;
	cv::Mat H1;
	cv::Mat H2;

	// void initialize
	void init( )
	{
		opencv_K = cv::Mat(3,3, CV_64F, K).clone();
		cv::transpose(opencv_K, opencv_K);
		
		if(! cv::invert(opencv_K, opencv_inverseK))
		{
			std::cout<< "warning: the K matrix is singular" << std::endl;
		}		

		opencv_inverseR = cv::Mat(3, 3, CV_64F, R).clone();
		opencv_R = cv::Mat(3,3, CV_64F, R).clone();
		cv::transpose(opencv_R, opencv_R);

		opencv_T = cv::Mat(3,1, CV_64F, T).clone();
		opencv_C = cv::Mat(3,1, CV_64F, C).clone();

		arraySize = w * h;
	
	}

	void init_relative(const ImageStruct &refImg)
	{
		opencv_relative_R = opencv_R * refImg.opencv_inverseR;
		opencv_relative_T = opencv_R * (opencv_C - refImg.opencv_C);	

		H1 = opencv_K * opencv_relative_R * refImg.opencv_inverseK;
		cv::Mat normalVector = (cv::Mat_<double>(1,3) << 0, 0, 1);
		H2 = opencv_K * opencv_relative_T * normalVector * refImg.opencv_inverseK;

		//cv::Mat H = _imgStruct_2[imageId].opencv_K * (_imgStruct_2[imageId].opencv_relative_R - _imgStruct_2[imageId].opencv_relative_T * normalVector / depth) * _imgStruct_1[0].opencv_inverseK;
		// cv::Mat H = H1 - H2/depth;
		//cv::Mat opencv_R = _imgStruct_2[imageId].opencv_R *_imgStruct_1[0].opencv_inverseR ;
		//cv::Mat opencv_T = _imgStruct_2[imageId].opencv_R * (_imgStruct_2[imageId].opencv_C - _imgStruct_1[0].opencv_C);	
		//cv::Mat H = _imgStruct_2[imageId].opencv_K * (opencv_R - opencv_T * normalVector / depth) * _imgStruct_1[0].opencv_inverseK;
	}

};


class pixelColor
{
public:
	cv::Mat _color;

	pixelColor()
	{
		//_color.create(4, 1, CV_64F)
		_color = (cv::Mat_<double>(4, 1) << 0., 0., 0., 0.0);
	}
	pixelColor(double r, double g, double b)
	{
		_color = (cv::Mat_<double>(4, 1) << r, g, b, 0.0);
	}
	pixelColor(const pixelColor &p)
	{
		_color = p._color;
	}

	pixelColor& operator+=(const pixelColor & p)
	{
		this->_color = this->_color + p._color;
		this->_color.at<double>(3) = 0.0;
		return *this;
	}

	pixelColor operator+(const pixelColor & p) const
	{
		pixelColor p1;
		p1._color = this->_color + p._color;
		p1._color.at<double>(3) = 0.0;
		return p1; 
	}

	pixelColor operator*(double num) const
	{
		pixelColor p1;
		p1._color = this->_color * num;
		p1._color.at<double>(3) = 0.0;
		return p1;
	}
	pixelColor operator*(const pixelColor &p) const
	{
		pixelColor p1;
		p1._color = this->_color.mul(p._color);
		return p1;
	}

	pixelColor operator-(const pixelColor &p) const
	{
		pixelColor p1;
		p1._color = this->_color - p._color;
		p1._color.at<double>(3) = 0.0;
		return p1; 
	}

	void sqrtRoot()
	{
		for(int i = 0; i<3; i++)
			this->_color.at<double>(i) = sqrt(this->_color.at<double>(i));
	}

};


#endif