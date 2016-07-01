#ifndef GIMAGE_H
#define GIMAGE_H

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "timer.h"
#include "array.h"

namespace gimage {

	struct Size {
		int width = 0;
		int height = 0;
	};

	enum class InterpType {
		BILINEAR
	};

	/**
	* Adds two arrays together. 
	* @param T1 the first array.
	* @param T2 the second array.
	* @param output the result of the addition.
	*/
	void GIMAGE_EXPORT add(Array& T1, Array& T2, Array& output);

	/**
	* Subtracts two arrays. Note this will perform T2 - T1.
	* @param T1 and first array.
	* @param T2 the second array.
	* @param output the result of the subtraction.
	*/
	void GIMAGE_EXPORT subtract(Array& T1, Array& T2, Array& output);

	/**
	* Resizes the input array to the new size of the output array.
	* @param input the input array to resize
	* @param output the output array, this will define the size of the output array.
	* @param type the interpolation algorithm to use.
	*/
	void GIMAGE_EXPORT resize(Array& input, Array& output, InterpType type);

	/**
	* Converts an 8 bit RGB image to grayscale. 
	* @param red the red channel of the image to convert.
	* @param green the green channel of the image to convert.
	* @param blue the blue channel of the image to convert.
	* @param output the output gray scale image. 
	*/
	void GIMAGE_EXPORT rgbToGray(ArrayUint8& red, ArrayUint8& green, ArrayUint8& blue, ArrayUint8& output);

	/**
	* Thresholds a given image and puts the result in the output.
	* @param input the image to threshold.
	* @param output the resulting thresholded image.
	* @param threshold the threshold. 
	*/
	void GIMAGE_EXPORT threshold(Array& input, Array& output, int threshold);

	/**
	* Performs convolution on the input image using the laplacian of a the gaussian blur operator.
	* @param input input image, can be 16 bit, 8 bit or double
	* @param output double array output. Note that this array will contain negative values.
	* @param sigma, sigma for the LOG
	* @param numRows number of rows in the input and output image.
	* @param numCols the number of columns in the input and output image.
	* @param logSize size of the kernel (must be odd).
	*/
	void GIMAGE_EXPORT laplacianOfGaussian(Array& input, DoubleArray& output, float sigma, int numRows, int numCols, int logSize);

	/**
	* Performs a Gaussian blur on a given image and stores it in the output.
	* @param input the input image
	* @param output the output image.
	* @param numRows the number of rows in the input image.
	* @param numCols the number of columns int he input image.
	* @param blurSize the size of the blur. This must be odd. Note that the blur filter will be square.
	*/
	void GIMAGE_EXPORT gaussianBlur(Array& input, Array& output, float sigma, int numRows, int numCols, int blurSize);
	/**
	* Performs the look up table method of window and leveling on the given image and stores the result in out.
	* @param input the input image.
	* @param out the output image.
	* @param numRows the number of rows in the image.
	* @param numCols the number of columns in the image.
	* @param window the window to use in the calculation.
	* @param level the level to use in the calculation.
	*/
	void GIMAGE_EXPORT windowAndLevel(Array& input, Array& out, int numRows, int numCols, int window, int level);
	/**
	* Performs canny edge detection on the input and outputs an image with only the edges in the output.
	* @param input the input image.
	* @param output the output image (edges only).
	* @param numRows number of rows in the image.
	* @param numCols the number of the columns in the image.
	* @param sigma sigma for the gaussian blur.
	* @param uint16_t lowerThresh lower threshold for the canny edge detector.
	* @param uint16_t upterThresh upper threshold for the canny edge detector.
	*/
	void GIMAGE_EXPORT cannyEdgeDetector(Array& input, Array &output, int numRows, int numCols,
										float sigma, int lowerThresh, int upperThresh);
}

#endif //GIMAGE_H