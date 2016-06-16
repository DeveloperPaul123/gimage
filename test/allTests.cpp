#include <cstdint>
#include "gimage.h"
#include "array.h"
#include "gtest\gtest.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

TEST(gimage, array_test) {
	gimage::DoubleArray d(1, 5);
	for (int i = 0; i < 5; i++) {
		d.setData<double>(0, i, (double)i);
	}
	double *da = static_cast<double*>(d.hostData());
	EXPECT_EQ(da[0], d.at<double>(0, 0));
}

TEST(gimage, gaussian__Test) {
	int blurSize = 57;
	cv::Mat input = cv::imread("test.tif", CV_16U);
	gimage::ArrayUint16 rawImage(input.rows, input.cols);
	gimage::ArrayUint16 output(input.rows, input.cols);
	uint16_t* raw_input = static_cast<uint16_t*>(rawImage.hostData());
	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j){
			uint16_t value = input.at<uint16_t>(i, j);
			rawImage.setData<uint16_t>(i, j, value);
		}
	}
	gimage::gaussianBlur(rawImage, output, 1.9f, input.rows, input.cols, blurSize);
	cv::Mat result(input.rows, input.cols, CV_16U, 
		static_cast<uint16_t*>(output.hostData()), cv::Mat::AUTO_STEP);
	cv::Size s(blurSize, blurSize);
	cv::Mat cvOut;
	cv::GaussianBlur(input, cvOut, s, 2.0, 2.0);
	cv::Mat dif;
	cv::subtract(result, cvOut, dif);
	uint16_t diffSum = 0;
	for (size_t r = 0; r < dif.rows; r++) {
		for (size_t c = 0; c < dif.cols; c++) {
			uint16_t value = dif.at<uint16_t>(r, c);
			diffSum += std::abs(value);
		}
	}

	cv::imshow("Input", input);
	cv::imshow("Reference", cvOut);
	cv::imshow("gimage", result);
	cv::waitKey(0);
	EXPECT_NEAR(diffSum, 0, 30);
}

TEST(gimage, window_level_Test) {
	cv::Mat input = cv::imread("test.tif", CV_16U);
	gimage::ArrayUint16 rawImage(input.rows, input.cols);
	gimage::ArrayUint16 out(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j){
			uint16_t value = input.at<uint16_t>(i, j);
			rawImage.setData<uint16_t>(i, j, value);
		}
	}
	gimage::windowAndLevel(rawImage, out, input.rows, input.cols, 32012, 652);
	cv::Mat result(input.rows, input.cols, CV_16U, 
		static_cast<uint16_t*>(out.hostData()), cv::Mat::AUTO_STEP);
	cv::imshow("Input", input);
	cv::imshow("Output", result);
	cv::waitKey(0);
}

TEST(gimage, matrix_mult_test) {
	gimage::MatrixD a(2, 2);
	gimage::MatrixD b(2, 2);
	for (int r = 0; r < a.rows; r++) {
		for (int c = 0; c < a.cols; c++) {
			a.setData<double>(r, c, (double)(2.0));
			b.setData<double>(r, c, (double)(4.0));
		}
	}
	gimage::MatrixD out = a*b;
	EXPECT_DOUBLE_EQ(out.at<double>(0, 0), 8.0);
}

TEST(gimage, canny_test) {
	cv::Mat input = cv::imread("test.tif", CV_16U);
	gimage::ArrayUint16 rawImage(input.rows, input.cols);
	gimage::ArrayUint16 out(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j){
			uint16_t value = input.at<uint16_t>(i, j);
			rawImage.setData<uint16_t>(i, j, value);
		}
	}
	gimage::cannyEdgeDetector(rawImage, out, input.rows, input.cols, 1.4f, 20, 300);
	cv::Mat result(input.rows, input.cols, CV_16U,
		static_cast<uint16_t*>(out.hostData()), cv::Mat::AUTO_STEP);
	cv::imshow("Input", input);
	cv::imshow("Output", result);
	cv::waitKey(0);
	cv::imwrite("canny_out.tif", result);
}

