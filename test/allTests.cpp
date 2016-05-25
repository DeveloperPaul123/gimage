#include <cstdint>
#include "gimage.h"
#include "array.h"
#include "gtest\gtest.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

TEST(gimage, gaussian__Test) {
	int blurSize = 57;
	cv::Mat input = cv::imread("test.tif", CV_16U);
	uint16_t *rawImg = new uint16_t[input.rows*input.cols];
	uint16_t *outputImg = new uint16_t[input.rows*input.cols];
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j){
			rawImg[i*input.cols + j] = input.at<uint16_t>(i, j);
		}
	}
	gimage::gaussianBlur(rawImg, outputImg, 1.9f, input.rows, input.cols, blurSize);
	cv::Mat result(input.rows, input.cols, CV_16U, outputImg, cv::Mat::AUTO_STEP);
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
	cv::imshow("Difference", dif);
	cv::waitKey(0);
	EXPECT_NEAR(diffSum, 0, 30);
}

TEST(gimage, windowLevel_Test) {
	cv::Mat input = cv::imread("test.tif", CV_16U);
	uint16_t *rawImg = new uint16_t[input.rows*input.cols];
	uint16_t *outputImg = new uint16_t[input.rows*input.cols];
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j){
			rawImg[i*input.cols + j] = input.at<uint16_t>(i, j);
		}
	}
	gimage::windowAndLevel(rawImg, outputImg, input.rows, input.cols, 32012, 652);
	cv::Mat result(input.rows, input.cols, CV_16U, outputImg, cv::Mat::AUTO_STEP);
	cv::imshow("Input", input);
	cv::imshow("Output", result);
	cv::waitKey(0);
}

TEST(gimage, input_array_test) {
	size_t size = 3;
	gimage::InputArray<uint16_t> a(size);
	for (int i = 0; i < size; i++) {
		a.setData((uint16_t)i, i);
	}
	EXPECT_EQ(a.get(0), 0);
	EXPECT_EQ(a.get(1), 1);
}

TEST(gimage, matrix_test) {
	gimage::MatrixUint16 a(3);
	for (int i = 0; i < 3; i++) {
		a.setData((uint16_t)i, i);
	}
	EXPECT_EQ(a.get(0), 0);
	EXPECT_EQ(a.get(1), 1);
	EXPECT_EQ(a.get(2), 2);
}

TEST(gimage, matrix_test_two) {
	gimage::MatrixUint8 a(2);
	for (int i = 0; i < 2; i++) {
		a.setData((uint8_t)i * 2, i);
	}
	EXPECT_EQ(a.get(0), 0);
	EXPECT_EQ(a.get(1), 2);
}

TEST(gimage, test_template_kernel) {
	gimage::MatrixUint8 a(2);
	gimage::MatrixUint8 b(2);
	gimage::MatrixUint8 out(2);
	for (int i = 0; i < 2; i++) {
		a.setData((uint8_t)i * 2, i);
		b.setData((uint8_t)i * 3, i);
	}

	gimage::test(a, b, out, 2);

	EXPECT_EQ(out.get(0), a.get(0) + b.get(0));
}

