#pragma once
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <deque>
namespace my {
	std::string getImageType(int number);
	void polyFit(cv::Mat& points, cv::Mat& x, double lambda);
	void polyFit(std::vector<cv::Point>& points, cv::Mat& x, double lambda);
	void drawPoly(cv::Mat& img, cv::Mat& paras, cv::Scalar& color);
	void drawPoly(cv::Mat& img, double para1, double para2, double para3, cv::Scalar& color);
	void drawLane(cv::Mat& img, std::vector<double> paraLeft, std::vector<double> paraRight);
	double deque_mean(std::deque<double>& q);
	void fileNameSeparater(const std::string& fileNameIn, std::string& fileNameOut, std::string& extension);
	int getDeviation(cv::Size size, const std::vector<double>& paraLeft, const std::vector<double>& paraRight);
}


