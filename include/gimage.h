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

	/**
	* Converts an 8 bit RGB image to grayscale. 
	* @param red the red channel of the image to convert.
	* @param green the green channel of the image to convert.
	* @param blue the blue channel of the image to convert.
	* @param output the output gray scale image. 
	*/
	void GIMAGE_EXPORT rgbToGray(Array& red, Array& green, Array& blue, Array& output);

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