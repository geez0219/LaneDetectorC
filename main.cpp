#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include "myFunction.h"

using namespace cv;
using namespace std;

//class myTrackBar {
//private:
//	static int sobelThresh;
//	static int LThresh;
//	static Mat imgL;
//	static Mat warpMatrix;
//public:
//	myTrackBar(Mat imgIn, Mat warpMatrixIn) {
//		sobelThresh = 15;
//		LThresh = 250;
//		Mat img_HSL[3], original_warp;
//		warpMatrix = warpMatrixIn;
//		medianBlur(imgIn, imgIn, 5);
//		cvtColor(imgIn, imgIn, COLOR_BGR2HLS);
//		split(imgIn, img_HSL);
//		imgL = img_HSL[1];
//		warpPerspective(imgL, original_warp, warpMatrix, imgIn.size());
//		imshow("original", original_warp);
//		namedWindow("test", WINDOW_NORMAL);
//		createTrackbar("sobel ", "test", &sobelThresh, 255, callBack);
//		createTrackbar("L ", "test", &LThresh, 255, callBack);
//		this->callBack(sobelThresh, 0);
//	}
//
//	static void callBack(int, void*) {
//		Mat img_HLS[3], imgL_warp, imgL_equal, sobelx, sobelxBool, LBool, imgOut[3], imgMerge;
//		// process sobel lane pixels
//		
//		Sobel(imgL, sobelx, CV_32F, 1, 1);
//		inRange(sobelx, Scalar(sobelThresh), Scalar(255), sobelxBool);
//		warpPerspective(sobelxBool, sobelxBool, warpMatrix, sobelxBool.size());
//		// process color lane pixels
//		warpPerspective(imgL, imgL_warp, warpMatrix, imgL.size());
//		equalizeHist(imgL_warp, imgL_equal);
//		inRange(imgL_equal, Scalar(LThresh), Scalar(255), LBool);
//		imgOut[0] = Mat::zeros(imgL.size(), CV_8UC1);
//		imgOut[1] = sobelxBool * 255;
//		imgOut[2] = LBool * 255;
//		merge(imgOut, 3, imgMerge);
//		imshow("test", imgMerge);
//	}
//};
//
//int myTrackBar::sobelThresh = 15;
//int myTrackBar::LThresh = 250;
//Mat myTrackBar::imgL;
//Mat myTrackBar::warpMatrix; 


int sizeXOrg = 1920, sizeYOrg = 1080;
int sizeXNew = 1920, sizeYNew = 1080;
Size orgSize(sizeXOrg, sizeYOrg);
Size newSize(sizeXNew, sizeYNew);
Size leftSize(sizeXOrg*0.7, sizeYOrg);
Size rightSize(sizeXOrg*0.3, sizeYOrg);

/* bob car*/
Point2f src[4] = { Point2f(0.42 * sizeXOrg, 0.53 * sizeYOrg),
				   Point2f(0.26 * sizeXOrg, 0.7 * sizeYOrg),
				   Point2f(0.7 * sizeXOrg, 0.7 * sizeYOrg),
				   Point2f(0.55 * sizeXOrg, 0.53 * sizeYOrg)};

//Point2f src[4] = { Point2f(0.41 * sizeXOrg, 0.67 * sizeYOrg),
//				   Point2f(0.20 * sizeXOrg, 0.9 * sizeYOrg),
//				   Point2f(0.85 * sizeXOrg, 0.9 * sizeYOrg),
//				   Point2f(0.61 * sizeXOrg, 0.67 * sizeYOrg) };

Point2f dst[4] = { Point2f(0 * sizeXNew, 0 * sizeYNew),
				   Point2f(0 * sizeXNew, 1 * sizeYNew),
				   Point2f(1 * sizeXNew, 1 * sizeYNew),
				   Point2f(1 * sizeXNew, 0 * sizeYNew)};


Mat warpMatrix = getPerspectiveTransform(src, dst);
Mat invWarpMatrix = getPerspectiveTransform(dst, src);

int thresholdL = 255;
int thresholdS = 80;
int thresholdSobel = 15; // 15

Scalar color[2] = { Scalar(255,0,0), Scalar(0,255,0) };
int midPoint = int(newSize.width / 2);
int yLow, yHigh;
int xHigh[2] = { midPoint - 1, newSize.width };
int xLow[2] = { 0, midPoint };
int windowNum = 16;
int smallerWindowLenX = newSize.width / 13;
int smallerWindowLenY = (float)newSize.height / windowNum;
int pointNumThresh = smallerWindowLenX * smallerWindowLenY * 0.1;
int predLaneThresh = 3;
int paraBufferSize = 15;
deque<double> leftRightParaQ1[2];
deque<double> leftRightParaQ2[2];
deque<double> leftRightParaQ3[2];

void getLeftRightPoints(Mat& img, Mat& allPoints, vector<Point> *leftRightPoints) {
	for (int j = 0; j < 2; j++) { // left and right
		for (int i = 0; i < windowNum; i++) {
			Mat selected, meanInWindow;
			vector<double> pointMean, pointStd;
			int pointNum;
			yHigh = (float)newSize.height / windowNum * (i + 1);
			yLow = (float)newSize.height / windowNum * i;
			// first stage window
			inRange(allPoints, Scalar(xLow[j], yLow), Scalar(xHigh[j], yHigh), selected);
			cv::meanStdDev(allPoints, pointMean, pointStd, selected);
			if (pointMean[0] == 0 || pointMean[1] == 0) continue;
			/*putText(laneP, "pointNum:" + to_string(pointNum) , Point(xLowLeft, yHigh), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
			putText(laneP, "std:" + to_string(pointStd[0]), Point(xLowLeft, yHigh-100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);*/
			// second stage window

			int newXLow = (pointMean[0] - smallerWindowLenX / 2 < 0 ? 0 : pointMean[0] - smallerWindowLenX / 2);
			int newXHigh = (pointMean[0] + smallerWindowLenX / 2 < 0 ? 0 : pointMean[0] + smallerWindowLenX / 2);
			int newYLow = (pointMean[1] - smallerWindowLenY / 2 < 0 ? 0 : pointMean[1] - smallerWindowLenY / 2);
			int newYHigh = (pointMean[1] + smallerWindowLenY / 2 < 0 ? 0 : pointMean[1] + smallerWindowLenY / 2);
			inRange(allPoints, Scalar(newXLow, newYLow), Scalar(newXHigh, newYHigh), selected);
			rectangle(img, Point(newXLow, newYLow), Point(newXHigh, newYHigh), color[j], 3);
			cv::meanStdDev(allPoints, pointMean, pointStd, selected);
			pointNum = countNonZero(selected);

			if (pointNum > pointNumThresh) {
				Point thePoint(pointMean[0], pointMean[1]);
				leftRightPoints[j].emplace_back(thePoint); // push_back to emplace_back
				circle(img, thePoint, 10, color[j]);
			}
		}
	}
}

void pipeLine(Mat& img) {
	Mat orgImg = img.clone();
	// first stage: median filter
	medianBlur(img, img, 5);  
	// convert color to HLS
	cvtColor(img, img, COLOR_BGR2HLS);
	Mat img_HLS[3], outputImg[3], SWarped,LWarped, LWarpedEqual, whiteLinePts, yellowLinePts, edgePts;
	split(img, img_HLS);

	/* detect edge */
	warpPerspective(img_HLS[1], LWarped, warpMatrix, newSize);
	Sobel(LWarped, edgePts, CV_32F, 1, 0);
	inRange(abs(edgePts), Scalar(thresholdSobel), Scalar(255), edgePts);
	/* detect yellow line*/
	warpPerspective(img_HLS[2], SWarped, warpMatrix, newSize);
	inRange(SWarped, Scalar(thresholdS), Scalar(255), yellowLinePts);
	/* detect white line*/
	equalizeHist(LWarped, LWarpedEqual);
	inRange(LWarpedEqual, Scalar(thresholdL), Scalar(255), whiteLinePts);
	Mat laneP = whiteLinePts | yellowLinePts | edgePts;

	/////// classify the all points to left or right ///////
	Mat allPoints;
	findNonZero(laneP, allPoints);
	cvtColor(laneP, laneP, COLOR_GRAY2BGR);

	vector<Point> leftRightPoints[2]; 
	getLeftRightPoints(laneP, allPoints, leftRightPoints);

	Mat leftRightPara[2];
	double lambda = 200000000000;// 200000000000

	for (int j = 0; j < 2; j++) {
		if (leftRightPoints[j].size() > predLaneThresh) {
			my::polyFit(leftRightPoints[j], leftRightPara[j], lambda);
			my::drawPoly(laneP, leftRightPara[j], color[j]); // temp
			if (leftRightParaQ1[j].size() < paraBufferSize) {
				leftRightParaQ1[j].push_back(leftRightPara[j].at<double>(0, 0));
				leftRightParaQ2[j].push_back(leftRightPara[j].at<double>(1, 0));
				leftRightParaQ3[j].push_back(leftRightPara[j].at<double>(2, 0));
			}
			else {
				leftRightParaQ1[j].pop_front();
				leftRightParaQ2[j].pop_front();
				leftRightParaQ3[j].pop_front();
				leftRightParaQ1[j].push_back(leftRightPara[j].at<double>(0, 0));
				leftRightParaQ2[j].push_back(leftRightPara[j].at<double>(1, 0));
				leftRightParaQ3[j].push_back(leftRightPara[j].at<double>(2, 0));
			}
		}
	}
	Mat imgRight;
	resize(laneP, imgRight, rightSize);
	vector<double> paraLeft = { my::deque_mean(leftRightParaQ1[0]), my::deque_mean(leftRightParaQ2[0]), my::deque_mean(leftRightParaQ3[0]) };
	vector<double> paraRight = { my::deque_mean(leftRightParaQ1[1]), my::deque_mean(leftRightParaQ2[1]), my::deque_mean(leftRightParaQ3[1]) };
	
	Mat laneImg(newSize, CV_8UC3, Scalar(0, 0, 0));
	
	my::drawLane(laneImg, paraLeft, paraRight);
	int deviation = my::getDeviation(newSize, paraLeft, paraRight);
	warpPerspective(laneImg, laneImg, invWarpMatrix, orgSize);

	Mat mask, invMask, orgROI, orgWithoutROI, newROI, imgLeft;
	inRange(laneImg, Scalar(0, 1, 0), Scalar(255, 255, 255), mask);
	bitwise_not(mask, invMask);
	bitwise_and(orgImg, orgImg, orgWithoutROI, invMask);
	bitwise_and(orgImg, orgImg, orgROI, mask);
	addWeighted(orgROI, 0.7, laneImg, 0.3, 0, newROI);
	imgLeft = newROI + orgWithoutROI;
	putText(imgLeft, "deviate:" + to_string(deviation), src[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
	resize(imgLeft, imgLeft, leftSize);
	hconcat(imgLeft, imgRight, img);
}

void pipeLine_resize(Mat& img) {
	Size targetSize(800, 600);
	resize(img, img, targetSize);
}


void pipeLine_drawROI(Mat& img) {
	vector<Point> points(4);
	for (int i = 0; i < 4; i++) {
		points[i] = Point(src[i].x, src[i].y);
	}

	polylines(img, points, 1, Scalar(255, 0, 0));

}

void pipeLine_lanePixel(Mat& img) {
	Mat orgImg = img.clone();
	// first stage: median filter
	medianBlur(img, img, 5);
	// convert color to HLS
	cvtColor(img, img, COLOR_BGR2HLS);
	Mat img_HLS[3], outputImg[3], SWarped, LWarped, LWarpedEqual, whiteLinePts, yellowLinePts, edgePts;
	split(img, img_HLS);

	/* detect edge */
	warpPerspective(img_HLS[1], LWarped, warpMatrix, newSize);
	Sobel(LWarped, edgePts, CV_32F, 1, 0);
	inRange(abs(edgePts), Scalar(thresholdSobel), Scalar(255), edgePts);

	/* detect yellow line*/
	warpPerspective(img_HLS[2], SWarped, warpMatrix, newSize);
	inRange(SWarped, Scalar(thresholdS), Scalar(255), yellowLinePts);

	/* detect white line*/
	equalizeHist(LWarped, LWarpedEqual);
	inRange(LWarpedEqual, Scalar(thresholdL), Scalar(255), whiteLinePts);
	Mat mask = whiteLinePts | yellowLinePts | edgePts;
	outputImg[0] = whiteLinePts;
	outputImg[1] = yellowLinePts;
	outputImg[2] = edgePts;
	Mat Pts, invMask, imgWithout, imgBG, imgFG, imgFGAfter;
	merge(outputImg, 3, Pts);

	img = Pts;

	warpPerspective(Pts, Pts, invWarpMatrix, orgSize);
	warpPerspective(mask, mask, invWarpMatrix, orgSize);
	bitwise_not(mask, invMask);
	bitwise_and(orgImg, orgImg, imgBG, invMask);
	bitwise_and(orgImg, orgImg, imgFG, mask);
	addWeighted(imgFG, 0.7, Pts, 0.3, 0, imgFGAfter);
	img = imgBG + imgFGAfter;
}

int main() {
	/* video processing */
	VideoCapture videoIn;
	VideoWriter videoOut;
	string videoDir = "lane_vid/";
	//vector<string> stringList = {"video2.avi"};
	vector<string> stringList = { "video5.mp4"};
	int idx = 1;
	for (auto &i: stringList) {
		cout << "processing file#" << idx << endl;
		string fileInName = videoDir + i;
		videoIn.open(fileInName.c_str());
		string fileOutName = "video" + to_string(5) + "output" + ".avi";
		videoOut.open(fileOutName.c_str(), VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, orgSize);
		if (!videoIn.isOpened()) {
			cout << "cannot open the video file" << endl;
			system("pause");
			exit(1);
		}
		Mat videoFrame;
		int counter = 0;
		while (true) {
			cout << "the counter:" << counter++ << endl;
			videoIn >> videoFrame;
			if (videoFrame.empty()) break;
			pipeLine(videoFrame);
			videoOut << videoFrame;
			namedWindow("test", WINDOW_NORMAL);
			imshow("test", videoFrame);
			waitKey(30);
		}
	}
	
	return 0;
}