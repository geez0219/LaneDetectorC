#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "myFunction.h"

std::string my::getImageType(int number)
{
	// find type
	int imgTypeInt = number % 8;
	std::string imgTypeString;

	switch (imgTypeInt)
	{
	case 0:
		imgTypeString = "8U";
		break;
	case 1:
		imgTypeString = "8S";
		break;
	case 2:
		imgTypeString = "16U";
		break;
	case 3:
		imgTypeString = "16S";
		break;
	case 4:
		imgTypeString = "32S";
		break;
	case 5:
		imgTypeString = "32F";
		break;
	case 6:
		imgTypeString = "64F";
		break;
	default:
		break;
	}

	// find channel
	int channel = (number / 8) + 1;

	return imgTypeString + "C" + std::to_string(channel);
}

void  my::polyFit(cv::Mat& points, cv::Mat& x, double lambda) {
	points.convertTo(points, CV_32F);
	cv::Mat b = points.col(1);
	cv::Mat A(points.rows, 3, CV_32F);
	cv::Mat regMatrix(3, 3, CV_32F);
	regMatrix.at<float>(0, 0) = lambda;

	A.col(2) = cv::Mat::ones(points.rows, 1, CV_32F);
	A.col(1) = points.col(0) * 1;
	cv::pow(A.col(1), 2, A.col(0));
	cv::solve(A.t()*A + regMatrix, A.t()*b, x, cv::DECOMP_SVD);
}

void my::polyFit(std::vector<cv::Point>& points, cv::Mat& x, double lambda) {
	cv::Mat A(points.size(), 3, CV_64F);
	cv::Mat b(points.size(), 1, CV_64F);
	for (int i = 0; i < points.size(); i++) {
		double x = points[i].x;
		double y = points[i].y;
		A.at<double>(i, 0) = y*y;
		A.at<double>(i, 1) = y;
		A.at<double>(i, 2) = 1;
		b.at<double>(i, 0) = x;
	}

	cv::Mat regMatrix = cv::Mat::zeros(3, 3, CV_64F);
	regMatrix.at<double>(0, 0) = lambda;
	cv::solve(A.t()*A + regMatrix, A.t()*b, x, cv::DECOMP_SVD);
}


void my::drawPoly(cv::Mat& img, cv::Mat& paras, cv::Scalar& color) {
	cv::Mat drawPoints(img.rows, 2, CV_32S);
	for (int i = 0; i < img.rows; i++) {
		drawPoints.at<int>(i, 0) = int(i*i*paras.at<double>(0, 0) + i*paras.at<double>(1, 0) + paras.at<double>(2, 0));
		drawPoints.at<int>(i, 1) = int(i);
	}
	cv::polylines(img, drawPoints, 0, color);
}


void my::drawPoly(cv::Mat& img, double para1, double para2, double para3, cv::Scalar& color) {
	cv::Mat drawPoints(img.rows, 2, CV_32S);
	for (int i = 0; i < img.rows; i++) {
		drawPoints.at<int>(i, 0) = int(i*i*para1 + i*para2 + para3);
		drawPoints.at<int>(i, 1) = int(i);
	}
	cv::polylines(img, drawPoints, 0, color);
}


void my::drawLane(cv::Mat& img, std::vector<double> paraLeft, std::vector<double> paraRight) {
	std::vector<cv::Point> leftLane(img.rows);
	std::vector<cv::Point> rightLane(img.rows);
	std::vector<std::vector<cv::Point>> countour(1);

	for (int i = 0; i < img.rows; i++) {
		int y = i;
		int xLeft = paraLeft[0] * i*i + paraLeft[1] * i + paraLeft[2];
		int xRight = paraRight[0] * i*i + paraRight[1] * i + paraRight[2];
		leftLane[i] = cv::Point(xLeft, y);
		rightLane[img.rows-1-i] = cv::Point(xRight, y);
	}

	cv::polylines(img, leftLane, 0, cv::Scalar(255, 0, 0));
	cv::polylines(img, rightLane, 0, cv::Scalar(0, 255, 0));
	countour[0].insert(countour[0].end(), leftLane.begin(), leftLane.end());
	countour[0].insert(countour[0].end(), rightLane.begin(), rightLane.end());
	cv::fillPoly(img, countour, cv::Scalar(0, 0, 255));
}

double my::deque_mean(std::deque<double>& q) {
	double sum = 0;
	for (auto &i : q) {
		sum += i;
	}
	return sum / q.size();
}


void my::fileNameSeparater(const std::string& fileNameIn, std::string& fileNameOut, std::string& extension) {
	auto pos = fileNameIn.find(".");
	if (pos == std::string::npos) {
		std::cout << "In fileNameSparater: cannot find the \" .\"" << std::endl;
		std::cin.get();
		exit(1);
	}
	if (pos == fileNameIn.size() - 1) {
		std::cout << "In fileNameSparater: the \".\" is the last char of string" << std::endl;
		std::cin.get();
		exit(1);
	}
	fileNameOut = fileNameIn.substr(0, pos);
	extension = fileNameIn.substr(pos + 1, std::string::npos);
}

int my::getDeviation(cv::Size size, const std::vector<double>& paraLeft, const std::vector<double>& paraRight) {
	int middleY = size.height / 2;
	int left = paraLeft[0] * middleY * middleY + paraLeft[1] * middleY + paraLeft[2];
	int right = paraRight[0] * middleY * middleY + paraRight[1] * middleY + paraRight[2];
	return right + left - size.width;
}