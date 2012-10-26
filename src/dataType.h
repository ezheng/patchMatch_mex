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
	pixelPos(cv::Mat pt)
	{
		_pt = pt;
	}

	void sub2Idx(double &ind_r, double &ind_g, double &ind_b, const double &height, const double &width) const
	{
		ind_r = _pt.at<double>(0) * height + _pt.at<double>(1);
		ind_g = ind_r + height * width;
		ind_b = ind_g + height * width;
	}
	double sub2Idx(const double &height) const
	{
		return _pt.at<double>(0) * height + _pt.at<double>(1);
	}
	double sub2Idx(const double &height, const double &width, int level) const
	{
		return width * height * level + _pt.at<double>(0) * height + _pt.at<double>(1);
	}

	void sub2Idx(std::vector<double> &weight, std::vector<double> &ind_r, std::vector<double> &ind_g, std::vector<double> &ind_b, const double &height, const double &width) const
	{
		double x_low = floor( _pt.at<double>(0));
		double x_heigh = ceil( _pt.at<double>(0));
		double y_low = floor( _pt.at<double>(1));
		double y_heigh = ceil( _pt.at<double>(1));

		weight.resize(4);
		weight[0] = (x_heigh - _pt.at<double>(0)) * (y_heigh - _pt.at<double>(1));
		weight[1] = (_pt.at<double>(0) - x_low)    * (y_heigh - _pt.at<double>(1));
		weight[2] = (x_heigh - _pt.at<double>(0)) * ( _pt.at<double>(1) - y_low );
		weight[3] = (_pt.at<double>(0) - x_low)    * ( _pt.at<double>(1) - y_low );

		ind_r.resize(4); ind_g.resize(4); ind_b.resize(4);
		double ind_r_int, ind_g_int, ind_b_int;
		pixelPos pt(x_low, y_low);
		pt._pt.at<double>(0) = x_low;	pt._pt.at<double>(1) = y_low;
		pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, width);
		ind_r[0] = ind_r_int;
		ind_g[0] = ind_g_int;
		ind_b[0] = ind_b_int;
		
		pt._pt.at<double>(0) = x_low;	pt._pt.at<double>(1) = y_heigh;
		pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, width);
		ind_r[1] = ind_r_int;
		ind_g[1] = ind_g_int;
		ind_b[1] = ind_b_int;

		pt._pt.at<double>(0) = x_heigh;	pt._pt.at<double>(1) = y_low;
		pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, width);
		ind_r[2] = ind_r_int;
		ind_g[2] = ind_g_int;
		ind_b[2] = ind_b_int;

		pt._pt.at<double>(0) = x_heigh;	pt._pt.at<double>(1) = y_heigh;
		pt.sub2Idx(ind_r_int, ind_g_int, ind_b_int, height, width);
		ind_r[3] = ind_r_int;
		ind_g[3] = ind_g_int;
		ind_b[3] = ind_b_int;
	}
};


class ImageStruct
{
public:
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

	// 
	cv::Mat opencv_K;
	cv::Mat opencv_R;
	cv::Mat opencv_T;
	cv::Mat opencv_C;

	cv::Mat opencv_inverseK;
	cv::Mat opencv_inverseR;

	// void initialize
	void init()
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


};


#endif