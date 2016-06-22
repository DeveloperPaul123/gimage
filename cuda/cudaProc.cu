#include "gimage.h"
#include "array.h"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define PI 3.14159265359
#define PRINT_INFO 1
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

/**
* Template function for checking cuda errors. 
*/
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

/**
* Helper function for cuda memory allocation. Simplifies calls to cudaMalloc.
* @param d_p device pointer.
* @param size_t element the number of elements. 
*/
template<typename T>
cudaError_t cudaAlloc(T*& d_p, size_t elements)
{
	return cudaMalloc((void**)&d_p, elements * sizeof(T));
}

/**
* Helper function to select the proper CUDA device based on memory.
* @return int the device index to use.
*/
int selectDevice() {
	int devices;
	cudaGetDeviceCount(&devices);
	if (devices > 1) {
		//need to select a device. 
		int bestDevice = -1;
		int maxMemory = -INT_MAX;
		for (int i = 0; i < devices; i++) {
			struct cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, i);
			int mem = properties.totalGlobalMem;
			if (mem > maxMemory) {
				maxMemory = mem;
				bestDevice = i;
			}
		}
		if (bestDevice != -1) {
			return bestDevice;
		}
		else {
			return 0;
		}
	}
	else {
		return 0;
	}
}

/**
* Confirms an RBG image to gray scale using the luminosity formula. 
* @param red the red channel
* @param green the green channel
* @param blue the blue channel.
* @param gray the gray scale output.
* @param numRows the number of rows in all the images.
* @param numCols the number of columns in all of the images.
*/
template<typename T>
__global__ void colorToGrey(T* red, T* green, T* blue, T* gray, int numRows, int numCols) {
	
	//get row and column in blcok
	int r = threadIdx.y + blockIdx.y*blockDim.y;
	int c = threadIdx.x + blockIdx.x*blockDim.x;
	//get unique point in image by finding position in grid.
	int index = c + r*blockDim.x*gridDim.x;
	int totalSize = numRows*numCols;
	if (index < totalSize) {
		double r = static_cast<double>(red[index]);
		double g = static_cast<double>(green[index]);
		double b = static_cast<double>(blue[index]);
		double grey = 0.21*r + 0.72*g + 0.07*b;
		gray[index] = static_cast<T>(grey);
	}
}

/**
* Gaussian blur kernal. Reads in a 16 bit image and outputs the blured image.
* @param d_in device input image.
* @param d_out device output image.
* @param filter gaussian filter array.
* @param numRows number of rows in the image.
* @param numCols number of cols in the image.
* @param blurSize the size of the blur.
*/
template<typename T>
__global__ void gaussian(T *d_in, T *d_out, const float* const filter, int numRows, int numCols, int blurSize) {
	//so filter width defines width of the filter.
	assert(blurSize % 2 == 1); //filter size should be odd.
	//get row and column in blcok
	int r = threadIdx.y + blockIdx.y*blockDim.y;
	int c = threadIdx.x + blockIdx.x*blockDim.x;
	//get unique point in image by finding position in grid.
	int offset = c + r*blockDim.x*gridDim.x;

	//holder for result of filtering. Hold it as a float for calculations. 
	float result = 0.0f;
	//check we don't access memory that doesn't exist. 
	if (offset >= numRows*numCols) {
		return;
	}
	//apply the filter. 
	for (int filter_r = -blurSize / 2; filter_r <= blurSize / 2; ++filter_r) {
		for (int filter_c = -blurSize / 2; filter_c <= blurSize / 2; ++filter_c) {
			//Find the global image position for this filter position
			//clamp to boundary of the image
			int rowCompare = r + filter_r >= 0 ? r + filter_r : 0;
			int colCompare = c + filter_c >= 0 ? c + filter_c : 0;
			//make sure we don't index rows and columns that don't exist. 
			int image_r = rowCompare <= static_cast<int>(numRows - 1) ? rowCompare : static_cast<int>(numRows - 1);
			int image_c = colCompare <= static_cast<int>(numCols - 1) ? colCompare : static_cast<int>(numCols - 1);

			float image_value = static_cast<float>(d_in[image_r * numCols + image_c]);
			float filter_value = filter[(filter_r + blurSize / 2) * blurSize + filter_c + blurSize / 2];
			//add filter value to result.
			result += image_value*filter_value;
		}
	}
	//set the output value. 
	d_out[offset] = static_cast<T>(result);
}


/**
* Generates a look up table used during window and leveling. 
* @param d_LUT device lookup table array. 
* @param window the window to use.
* @param level the level to use.
* @param levels the number of levels (this is also the size of d_LUT)
*/
__global__ void generateLUT(int* d_LUT, const int window, const int level, const int levels) {
	int a, b;
	int halfwin = window / 2;
	a = level - halfwin;
	if (a < 1) a = 1;
	b = level + halfwin;
	if (b > levels) b = levels;
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < levels) {
		if (id < a) {
			d_LUT[id] = 0;
		}
		else if (id >= a && id <= b) {
			d_LUT[id] = ((levels / window)*(id - a));
		}
		else {
			d_LUT[id] = levels;
		}
	}
	
}

/**
* Performs window and leveling on a given image and stores result in the output. 
* @param input the input image. 
* @param output the output image. 
* @param d_LUT the device look up table. 
* @param numRows the number of rows in the image. 
* @param numCols the number of columns in the image. 
* @param window the window to use in the calculation.
* @param level the level to use in the calculation.
* @param levels the number of levels in the image and LUT
*/
template<typename T>
__global__ void cudaWindowLevel(T* input, T *output, int *d_LUT, int numRows, int numCols, int window, int level, int levels) {
	//get row and column in blcok
	int r = threadIdx.y + blockIdx.y*blockDim.y;
	int c = threadIdx.x + blockIdx.x*blockDim.x;
	//get unique point in image by finding position in grid.
	int index = c + r*blockDim.x*gridDim.x;
	if (index < numRows*numCols) {
		T in = input[index];
		if (in < levels) {
			int out = d_LUT[in];
			output[index] = (T)out;
		}	
	}
}

/**
* Calculates the gradient values and directions for a given input. Stores angles in d_theta and the gradient in d_gradient
* @param d_in the input image.
* @param d_gradient the array to store the gradient values.
* @param d_theta array to store our gradient directions. 
* @param k_gx the sobel operator in x (created on host)
* @param k_gy the sobel operator in y (created on host)
* @param numRows the number of rows in the image.
* @param numCols the number of columns in the image. 
*/
template<typename T>
__global__ void gradientAndDirection(T *d_in, T *d_gradient, int* d_theta, int* k_gx, int* k_gy, int numRows, int numCols) {
	//get row and column in the current grid (this should be a sub set of the image if it is large enough.
	int r = threadIdx.y + blockIdx.y*blockDim.y;
	int c = threadIdx.x + blockIdx.x*blockDim.x;
	//get unique point in image by finding position in grid.
	int index = c + r*blockDim.x*gridDim.x;

	if (index >= numRows*numCols) {
		return;
	}

	//run convolution on the image with the sobel filters. 
	float x_res = 0.0f;
	float y_res = 0.0f;
	int kernelSize = 3;
	//apply the filter. 
	for (int filter_r = -kernelSize / 2; filter_r <= kernelSize / 2; ++filter_r) {
		for (int filter_c = -kernelSize / 2; filter_c <= kernelSize / 2; ++filter_c) {
			//Find the global image position for this filter position
			//clamp to boundary of the image
			int rowCompare = r + filter_r >= 0 ? r + filter_r : 0;
			int colCompare = c + filter_c >= 0 ? c + filter_c : 0;
			//make sure we don't index rows and columns that don't exist. 
			int image_r = rowCompare <= static_cast<int>(numRows - 1) ? rowCompare : static_cast<int>(numRows - 1);
			int image_c = colCompare <= static_cast<int>(numCols - 1) ? colCompare : static_cast<int>(numCols - 1);

			float image_value = static_cast<float>(d_in[image_r * numCols + image_c]);
			float filter_x = static_cast<float>(k_gx[(filter_r + kernelSize / 2) * kernelSize + filter_c + kernelSize / 2]);
			float filter_y = static_cast<float>(k_gy[(filter_r + kernelSize / 2) * kernelSize + filter_c + kernelSize / 2]);
			//add filter value to result.
			x_res += image_value*filter_x;
			y_res += image_value*filter_y;
		}
	}

	//store the gradient magnitude. 
	d_gradient[index] = static_cast<T>(sqrtf(powf(x_res, 2.0f) + powf(y_res, 2.0f)));
	double angle = (atan2f(y_res, x_res)) / PI * 180.0;
	int correctAngle;
	/* Convert actual edge direction to approximate value */
	if (((angle < 22.5) && (angle > -22.5)) || (angle > 157.5) || (angle < -157.5))
		correctAngle = 0;
	if (((angle > 22.5) && (angle < 67.5)) || ((angle < -112.5) && (angle > -157.5)))
		correctAngle = 45;
	if (((angle > 67.5) && (angle < 112.5)) || ((angle < -67.5) && (angle > -112.5)))
		correctAngle = 90;
	if (((angle > 112.5) && (angle < 157.5)) || ((angle < -22.5) && (angle > -67.5)))
		correctAngle = 135;
	//store the angle. 
	d_theta[index] = correctAngle;
}

/**
* Performs non-maximum suppression to further isolate edges during canny edge detection. This will save edges that are local maxima. It will then check 
* if the value is greater than the upper threshold, between the upper and lower threshold or below the lower threshold. If it is above the upper threshold the value
* is labeled as definitely being part of an edge. If it is inbetween, it will only be counted as an edge if it is connected to a definite edge, and if it below the lower
* threhold, it will be discarded.
* @param d_gradMag device pointer to gradient magnitude array.
* @param d_theta device pointer to gradient direction array.
* @param d_out the output array with the final edges. 
* @param upperThresh the upper threshold to check against.
* @param lowerThresh the lower threshold to check against. 
* @param numRows the number of rows in the arrays (should be the same for all).
* @param numCols the number of columns in the arrays (should be the same for all).
*/
template<typename T>
__global__ void nonMaximumSuppression(T* d_gradMag, int* d_theta, T* d_out, int numRows, int numCols) {
	//get row and column in the current grid (this should be a sub set of the image if it is large enough.
	int r = threadIdx.x + blockIdx.x*blockDim.x;
	int c = threadIdx.y + blockIdx.y*blockDim.y;
	//get unique point in image by finding position in grid.
	int index = c + r*blockDim.x*gridDim.x;

	if (index >= numRows*numCols) {
		return;
	}

	T value = d_gradMag[index];
	int direction = d_theta[index];
	int fCheck, sCheck = -1;
	switch (direction) {
	case 0:
		//horizontal
		fCheck = r + (c-1)*blockDim.x*gridDim.x;
		sCheck = r + (c+1)*blockDim.x*gridDim.x;
		break;
	case 45:
		//one row less one column more
		fCheck = (r - 1) + (c+1)*blockDim.x*gridDim.x;
		sCheck = (r + 1) + (c-1)*blockDim.x*gridDim.x;
		break;
	case 90:
		//vertical
		fCheck = (r - 1) + c*blockDim.x*gridDim.x;
		sCheck = (r + 1) + c*blockDim.x*gridDim.x;
		break;
	case 135:
		fCheck = (r - 1) + (c -1)*blockDim.x*gridDim.x;
		sCheck = (r + 1) + (c + 1)*blockDim.x*gridDim.x;
		break;
	}

	if (fCheck < numRows*numCols && sCheck < numRows*numCols) {
		T v1 = d_gradMag[fCheck];
		T v2 = d_gradMag[sCheck];
		int maxIndex = -1;
		if (value > v1 && value > v2) {
			maxIndex = index;
		}
		else if (value < v1 && v1 > v2) {
			maxIndex = fCheck;
		}
		else if (value < v2 && v2 > v1){
			maxIndex = sCheck;
		}

		if (maxIndex > 0 && maxIndex < numRows*numCols) {
			d_out[maxIndex] = d_gradMag[maxIndex];	
		}
	}
}

/**
* Performs hysteresis thresholding using two thresholds. If a value is above the upper threshold then it is deemed to definitely be an edge.
* If it is between the two thresholds, then it is only considered an edge if it is connected to a definite edge. If it is below or eqaul to 
* the lower threshold then it is definitely not an edge. 
* @param d_in input gradient magnitudes after non maximum suppression.
* @param d_out output edge image.
* @param theta gradient directions
* @param upper the upper threshold
* @param lower the lower threshold
* @param numRows the number of rows in all the arrays
* @param numCols the number of columns in all the arrays
*/
template<typename T> 
__global__ void hysteresisThresholding(T* d_in, T* d_out, int* theta, int upper, int lower, int numRows, int numCols) {
	//get row and column in the current grid (this should be a sub set of the image if it is large enough.
	int r = threadIdx.x + blockIdx.x*blockDim.x;
	int c = threadIdx.y + blockIdx.y*blockDim.y;
	//get unique point in image by finding position in grid.
	int index = c + r*blockDim.x*gridDim.x;
	int totalSize = numRows*numCols;
	if (index < totalSize) {
		T value = d_in[index];
		if (static_cast<int>(value) >= upper) {
			d_out[index] = value;
		}
		else if (static_cast<int>(value) <= lower) {
			d_out[index] = static_cast<T>(0);
		}
		else {
			//inbetween both values so walk the path. 
			//get the direction.
			int totBlockSize = blockDim.x*gridDim.x;
			int direction = theta[index];
			int rowOffset, colOffset = 1;
			bool foundFirst = false;
			while (true) {
				//traverse the path.
				int idxOne = -1;
				switch (direction) {
				case 0:
					//traverse column wise. 
					idxOne = r + (c - colOffset)*totBlockSize;
					break;
				case 45:
					idxOne = (r - rowOffset) + (c + colOffset)*totBlockSize;
					break;
				case 90:
					idxOne = (r - rowOffset) + c*totBlockSize;
					break;
				case 135:
					idxOne = (r - rowOffset) + (c - colOffset)*totBlockSize;
					break;
				}

				if (idxOne < totalSize && idxOne >= 0) {
					T v1 = d_in[idxOne];
					int dirOne = theta[idxOne];
					int v1Cast = static_cast<int>(v1);

					if (v1Cast <= lower) {
						//below lower threshold so no good. 
						foundFirst = false;
						break;
					}
					else if (v1Cast >= upper) {
						//def an edge so we're good. 
						foundFirst = true;
						//go ahead and set the value.
						d_out[index] = value;
						break;
					}
					else if (dirOne != direction) {
						foundFirst = false;
						break;
					}
					else {
						//increment and continue.
						colOffset++;
						rowOffset++;
					}
				}
				else {
					if (r + rowOffset >= numRows || c + colOffset >= numCols) {
						foundFirst = false;
						break;
					}
				}
				

			}

			//reset offsets. 
			colOffset = 1;
			rowOffset = 1;

			if (!foundFirst) {
				while (true) {
					int idxTwo = -1;
					switch (direction) {
					case 0:
						//traverse column wise. 
						idxTwo = r + (c + colOffset)*totBlockSize;
						break;
					case 45:
						idxTwo = (r + rowOffset) + (c - colOffset)*totBlockSize;
						break;
					case 90:
						idxTwo = (r + rowOffset) + c*totBlockSize;
						break;
					case 135:
						idxTwo = (r + rowOffset) + (c + colOffset)*totBlockSize;
						break;
					}

					if (idxTwo < totalSize && idxTwo >= 0) {

						T v2 = d_in[idxTwo];
						int dirTwo = theta[idxTwo];
						int v2Cast = static_cast<int>(v2);

						if (v2Cast <= lower) {
							//below lower threshold so no good. 
							d_out[index] = static_cast<T>(0);
							break;
						}
						else if (v2Cast >= upper) {
							//def an edge so we're good. 
							d_out[index] = value;
							break;
						}
						else if (dirTwo != direction) {
							d_out[index] = static_cast<T>(0);
							break;
						}
						else {
							//increment and continue.
							colOffset++;
							rowOffset++;
						}
					}
					else {
						if (r + rowOffset >= numRows || c + colOffset >= numCols) {
							d_out[index] = static_cast<T>(0);
						}
					}
				}
			}	
		}
	}
}

template<typename T>
__global__ void houghCircles(T* d_in, T* d_accumalator, int radius, int numRows, int numCols) {
	//get row and column in the current grid (this should be a sub set of the image if it is large enough.
	int r = threadIdx.x + blockIdx.x*blockDim.x;
	int c = threadIdx.y + blockIdx.y*blockDim.y;
	//get unique point in image by finding position in grid.
	int index = c + r*blockDim.x*gridDim.x;
	int totalSize = numRows*numCols;
	if (index < totalSize) {
		//TODO: Finish implementing hough circles, should go through all the 
		//angles for a circle (i.e. 0 to 2pi) and then get a and b. Save these
		//"votes" in the accumulator matrix. Will find the maximums of this accumulator
		//matrix later. 
	}
}

/**
* Namespace for all gimage functions.
*/
namespace gimage {

	Array& Array::operator=(Array& other) {
		if (this == &other) {
			return *this;
		}
		return *this;
	}

	DoubleArray::DoubleArray(int rows, int cols) : Array(rows, cols, Type::DOUBLE) {
		allocate(size());
	}

	DoubleArray::DoubleArray(DoubleArray &other) : Array(other.rows, other.cols, Type::DOUBLE){
		allocate(other.size());
		double* oData = static_cast<double*>(other.hostData());
		std::memcpy(h_data, oData, totalSize());
	}

	DoubleArray::~DoubleArray() {
		if (h_data) {
			delete[] h_data;
		}
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}
	}

	Array& DoubleArray::operator+(Array &other) {
		assert(rows == other.rows && cols == other.cols);
		DoubleArray out(rows, cols);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				out.setData<double>(r, c, at<double>(r, c) +
					other.at<double>(r, c));
			}
		}

		return out;
	}

	Array& DoubleArray::operator-(Array &other) {
		assert(rows == other.rows && cols == other.cols);
		DoubleArray out(rows, cols);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				out.setData<double>(r, c, at<double>(r, c) -
					other.at<double>(r, c));
			}
		}

		return out;
	}

	Array& DoubleArray::operator=(Array &other) {
		if (this == &other) {
			return *this;
		}
		assert(other.getType() == getType());
		rows = other.rows;
		cols = other.cols;
		setSize(rows*cols);
		delete[] h_data;
		allocate(totalSize());
		double* otherData = static_cast<double*>(other.hostData());
		std::memcpy(h_data, otherData, totalSize());
		return *this;
	}

	void* DoubleArray::hostData() {
		return h_data;
	}

	void* DoubleArray::deviceData() {
		return d_data;
	}

	
	void DoubleArray::gpuAlloc() {
		if (!d_data) {
			checkCudaErrors(cudaAlloc(d_data, size()));
		}
	}

	void DoubleArray::gpuFree() {
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}	
	}

	int DoubleArray::totalSize() {
		return size() * sizeof(double);
	}

	void DoubleArray::clone(Array& other) {
		assert(other.getType() == getType());
		assert(other.size() == size());
		std::memcpy(static_cast<double*>(other.hostData()), 
			static_cast<double*>(hostData()), totalSize());
	}

	void DoubleArray::memcpy(MemcpyDirection dir) {
		if (dir == MemcpyDirection::HOST_TO_DEVICE) {
			checkCudaErrors(cudaMemcpy(d_data, h_data, totalSize(), cudaMemcpyHostToDevice));
		}
		else {
			checkCudaErrors(cudaMemcpy(h_data, d_data, totalSize(), cudaMemcpyDeviceToHost));
		}
	}

	void DoubleArray::allocate(int size) {
		h_data = new double[size];
	}

	ArrayUint16::ArrayUint16(int rows, int cols) : Array(rows, cols, Type::UINT16) {
		allocate(size());
	}

	ArrayUint16::ArrayUint16(ArrayUint16 &other) : Array(other.rows, other.cols, Type::UINT16){
		allocate(other.size());
		uint16_t* oData = static_cast<uint16_t*>(other.hostData());
		std::memcpy(h_data, oData, totalSize());
	}

	ArrayUint16::~ArrayUint16() {
		delete[] h_data;
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}
	}

	Array& ArrayUint16::operator=(Array &other) {
		if (this == &other) {
			return *this;
		}
		assert(other.getType() == getType());
		rows = other.rows;
		cols = other.cols;
		setSize(rows*cols);
		delete[] h_data;
		allocate(totalSize());
		uint16_t* otherData = static_cast<uint16_t*>(other.hostData());
		std::memcpy(h_data, otherData, totalSize());
		return *this;
	}

	Array& ArrayUint16::operator+(Array &other) {
		assert(rows == other.rows && cols == other.cols);
		ArrayUint16 out(rows, cols);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				out.setData<uint16_t>(r, c, at<uint16_t>(r, c) +
					other.at<uint16_t>(r, c));
			}
		}

		return out;
	}

	Array& ArrayUint16::operator-(Array &other) {
		assert(rows == other.rows && cols == other.cols);
		ArrayUint16 out(rows, cols);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				out.setData<uint16_t>(r, c, at<uint16_t>(r, c) -
					other.at<uint16_t>(r, c));
			}
		}

		return out;
	}

	void* ArrayUint16::hostData() {
		return h_data;
	}

	void* ArrayUint16::deviceData() {
		return d_data;
	}

	void ArrayUint16::gpuAlloc() {
		if (!d_data) {
			checkCudaErrors(cudaAlloc(d_data, size()));
		}
	}

	void ArrayUint16::gpuFree() {
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}	
	}

	int ArrayUint16::totalSize() {
		return size() * sizeof(uint16_t);
	}

	void ArrayUint16::memcpy(MemcpyDirection dir) {
		if (dir == MemcpyDirection::HOST_TO_DEVICE) {
			checkCudaErrors(cudaMemcpy(d_data, h_data, totalSize(), cudaMemcpyHostToDevice));
		}
		else {
			checkCudaErrors(cudaMemcpy(h_data, d_data, totalSize(), cudaMemcpyDeviceToHost));
		}
	}

	void ArrayUint16::clone(Array& other) {
		assert(other.getType() == getType());
		assert(other.size() == size());
		std::memcpy(static_cast<uint16_t*>(other.hostData()), 
			static_cast<uint16_t*>(hostData()), totalSize());
	}

	void ArrayUint16::allocate(int size) {
		h_data = new uint16_t[size];
	}

	ArrayUint8::ArrayUint8(int rows, int cols) : Array(rows, cols, Type::UINT8) {
		allocate(size());
	}

	ArrayUint8::ArrayUint8(ArrayUint8 &other) : Array(other.rows, other.cols, Type::UINT8){
		allocate(other.size());
		uint8_t* oData = static_cast<uint8_t*>(other.hostData());
		std::memcpy(h_data, oData, totalSize());
	}

	ArrayUint8::~ArrayUint8() {
		delete[] h_data;
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}
	}

	Array& ArrayUint8::operator+(Array &other) {
		assert(rows == other.rows && cols == other.cols);
		ArrayUint8 out(rows, cols);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				out.setData<uint8_t>(r, c, at<uint8_t>(r, c) +
					other.at<uint8_t>(r, c));
			}
		}

		return out;
	}

	Array& ArrayUint8::operator-(Array &other) {
		assert(rows == other.rows && cols == other.cols);
		ArrayUint8 out(rows, cols);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				out.setData<uint8_t>(r, c, at<uint8_t>(r, c) -
					other.at<uint8_t>(r, c));
			}
		}

		return out;
	}

	Array& ArrayUint8::operator=(Array &other) {
		if (this == &other) {
			return *this;
		}
		assert(other.getType() == getType());
		rows = other.rows;
		cols = other.cols;
		setSize(rows*cols);
		delete[] h_data;
		allocate(totalSize());
		uint8_t* otherData = static_cast<uint8_t*>(other.hostData());
		std::memcpy(h_data, otherData, totalSize());
		return *this;
	}

	void* ArrayUint8::hostData() {
		return h_data;
	}

	void* ArrayUint8::deviceData() {
		return d_data;
	}

	void ArrayUint8::gpuAlloc() {
		if (!d_data) {
			checkCudaErrors(cudaAlloc(d_data, size()));
		}
	}

	void ArrayUint8::gpuFree() {
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}
	}

	int ArrayUint8::totalSize() {
		return size() * sizeof(uint8_t);
	}

	void ArrayUint8::memcpy(MemcpyDirection dir) {
		if (dir == MemcpyDirection::HOST_TO_DEVICE) {
			checkCudaErrors(cudaMemcpy(d_data, h_data, totalSize(), cudaMemcpyHostToDevice));
		}
		else {
			checkCudaErrors(cudaMemcpy(h_data, d_data, totalSize(), cudaMemcpyDeviceToHost));
		}
	}

	void ArrayUint8::clone(Array& other) {
		assert(other.getType() == getType());
		assert(other.size() == size());
		std::memcpy(static_cast<uint8_t*>(other.hostData()),
			static_cast<uint8_t*>(hostData()), totalSize());
	}

	void ArrayUint8::allocate(int size) {
		h_data = new uint8_t[size];
	}

	MatrixD::MatrixD(int size) : DoubleArray(1, size) {
	}

	MatrixD::MatrixD(int rows, int cols) : DoubleArray(rows, cols) {
	}

	double MatrixD::det() {
		//TODO: calculate determinant
		return 1.0;
	}

	/**
	* Performs matrix multiplication. 
	*/
	MatrixD MatrixD::operator*(MatrixD other) {
		assert(cols == other.rows);
		MatrixD out(rows, other.cols);
	
		int outRows = out.rows;
		int outCols = out.cols;
		for (int r = 0; r < outRows; r++) {
			for (int c = 0; c < outCols; c++) {
				//find sum for this position. 
				double sum = 0.0;
				for (int i = 0; i < other.rows; i++) {
					sum += other.at<double>(i, c) * at<double>(r, i);
				}
				out.setData<double>(r, c, sum);
			}
		}
		return out;
	}

	/**
	* Converts a color image to a grayscale image.
	* @param
	*/
	void GIMAGE_EXPORT rgbToGray(ArrayUint8& red, ArrayUint8& green, ArrayUint8& blue, ArrayUint8& gray) {

		//allocate arrays and move data to device. 
		red.gpuAlloc();
		green.gpuAlloc();
		blue.gpuAlloc();
		gray.gpuAlloc();
		
		//move data to device.
		red.memcpy(MemcpyDirection::HOST_TO_DEVICE);
		green.memcpy(MemcpyDirection::HOST_TO_DEVICE);
		blue.memcpy(MemcpyDirection::HOST_TO_DEVICE);

		//set the gray image to nothing.
		checkCudaErrors(cudaMemset(static_cast<uint8_t*>(gray.deviceData()), 0, gray.totalSize()));
		int numRows = red.rows;
		int numCols = red.cols;

		//select the device
		int device = selectDevice();
		checkCudaErrors(cudaSetDevice(device));
		struct cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);
		//get max threads and threads per block.
		int maxThreadsPerBlock = properties.maxThreadsPerBlock;
		int threadsPerBlock = std::sqrt(maxThreadsPerBlock);
#if PRINT_INFO
		std::cout << "GPU: " << properties.name << std::endl;
		std::cout << "Using " << properties.multiProcessorCount << " multiprocessors" << std::endl;
		std::cout << "Max threads per block: " << properties.maxThreadsPerBlock << std::endl;
		std::cout << "Max grid size: " << properties.maxGridSize[0] << std::endl;
		std::cout << "Threads per block " << threadsPerBlock << std::endl;
#endif	

		//specify block size. 
		dim3 block_size(threadsPerBlock, threadsPerBlock);
		/*
		* Specify the grid size for the GPU.
		* Make it generalized, so that the size of grid changes according to the input image size
		*/
		dim3 grid_size;
		grid_size.x = (numCols + block_size.x - 1) / block_size.x;  /*< Greater than or equal to image width */
		grid_size.y = (numRows + block_size.y - 1) / block_size.y; /*< Greater than or equal to image height */
		
		//gpu timer for 
		GpuTimer timer;
		timer.Start();
		colorToGrey << <grid_size, block_size >> >(static_cast<uint8_t*>(red.deviceData()), static_cast<uint8_t*>(green.deviceData()),
			static_cast<uint8_t*>(blue.deviceData()), static_cast<uint8_t*>(gray.deviceData()), numRows, numCols);
		timer.Stop();
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError()); 
		//measure how long the kernel took
		float ms = timer.Elapsed();

#if PRINT_INFO
		printf("RGB to gray kernel took %f ms.\n", ms);
#endif
		//copy results back.
		gray.memcpy(MemcpyDirection::DEVICE_TO_HOST);

		//free gpu memory. 
		red.gpuFree();
		green.gpuFree();
		blue.gpuFree();
		gray.gpuFree();
	}

	/**
	* Performs a Gaussian blur on a given image and stores it in the output.
	* @param input the input image
	* @param output the output image.
	* @param numRows the number of rows in the input image.
	* @param numCols the number of columns int he input image.
	* @param blurSize the size of the blur. This must be odd. Note that the blur filter will be square.
	*/
	void GIMAGE_EXPORT gaussianBlur(Array& input, Array& output, float sigma, int numRows, int numCols, int blurSize) {
		//blur size must be odd. 
		assert(blurSize % 2 == 1);
		//first calculate the filter. 
		float *h_filter = new float[blurSize*blurSize];
		float filterSum = 0.f;

		for (int r = -blurSize / 2; r <= blurSize / 2; ++r) {
			for (int c = -blurSize / 2; c <= blurSize / 2; ++c) {
				float filterValue = expf(-(float)(c * c + r * r) / (2.f * sigma * sigma));
				h_filter[(r + blurSize / 2) * blurSize + c + blurSize / 2] = filterValue;
				filterSum += filterValue;
			}
		}

		float normalizationFactor = 1.f / filterSum;

		for (int r = -blurSize / 2; r <= blurSize / 2; ++r) {
			for (int c = -blurSize / 2; c <= blurSize / 2; ++c) {
				h_filter[(r + blurSize / 2) * blurSize + c + blurSize / 2] *= normalizationFactor;
			}
		}

		//select the device
		int device = selectDevice();
		checkCudaErrors(cudaSetDevice(device));
		struct cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);

		//now we can filter the image. 
		int size = input.totalSize();
		float *d_filter;
		int totalBlurSize = blurSize*blurSize;

		//allocated memory for filter. 
		checkCudaErrors(cudaAlloc(d_filter, totalBlurSize));
		checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*totalBlurSize, cudaMemcpyHostToDevice));

		//allocate image memory.
		input.gpuAlloc();
		output.gpuAlloc();
		//copy memory to device. 
		input.memcpy(MemcpyDirection::HOST_TO_DEVICE);

		int maxThreadsPerBlock = properties.maxThreadsPerBlock;
		int threadsPerBlock = std::sqrt(maxThreadsPerBlock);
#if PRINT_INFO
		std::cout << "GPU: " << properties.name << std::endl;
		std::cout << "Using " << properties.multiProcessorCount << " multiprocessors" << std::endl;
		std::cout << "Max threads per block: " << properties.maxThreadsPerBlock << std::endl;
		std::cout << "Max grid size: " << properties.maxGridSize[0] << std::endl;
		std::cout << "Threads per block " << threadsPerBlock << std::endl;
#endif		
		//specify block size. 
		dim3 block_size(threadsPerBlock, threadsPerBlock);
		/*
		* Specify the grid size for the GPU.
		* Make it generalized, so that the size of grid changes according to the input image size
		*/
		dim3 grid_size;
		grid_size.x = (numCols + block_size.x - 1) / block_size.x;  /*< Greater than or equal to image width */
		grid_size.y = (numRows + block_size.y - 1) / block_size.y; /*< Greater than or equal to image height */
#if PRINT_INFO
		std::cout << "Grid size: (" << grid_size.x << " , " << grid_size.y << ")" << std::endl;
#endif	
		//get image type.
		gimage::Type t = input.getType();
		GpuTimer timer;
		
		switch (t) {
		case Type::UINT16:	
			//call the kernal.
			timer.Start();
			gaussian << <grid_size, block_size >> >(static_cast<uint16_t*>(input.deviceData()), static_cast<uint16_t*>(output.deviceData()), 
				d_filter, numRows, numCols, blurSize);
			timer.Stop();
			break;
		case Type::DOUBLE:
			//call the kernal.
			timer.Start();
			gaussian << <grid_size, block_size >> >(static_cast<double*>(input.deviceData()), static_cast<double*>(output.deviceData()),
				d_filter, numRows, numCols, blurSize);
			timer.Stop();
			break;
		}

		float ms = timer.Elapsed();
#if PRINT_INFO
		printf("Kernel took %f ms\n", ms);
#endif
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		output.memcpy(MemcpyDirection::DEVICE_TO_HOST);

		//clean up gpu
		output.gpuFree();
		input.gpuFree();
		checkCudaErrors(cudaFree(d_filter));

		//clean up host
		delete[] h_filter;	
	}

	/**
	* Performs the look up table method of window and leveling on the given image and stores the result in out.
	* @param input the input image.
	* @param out the output image.
	* @param numRows the number of rows in the image.
	* @param numCols the number of columns in the image.
	* @param window the window to use in the calculation.
	* @param level the level to use in the calculation.
	*/
	void GIMAGE_EXPORT windowAndLevel(Array& input, Array& out, int numRows, int numCols, int window, int level) {
		
		//perform assertions
		assert(input.getType() == out.getType());
		assert(input.size() == out.size());
		assert(input.rows == out.rows && input.cols == out.cols && input.rows == numRows && input.cols == numCols);
		assert(window > 0 && level > 0);

		//select the device
		int device = selectDevice();
		checkCudaErrors(cudaSetDevice(device));
		struct cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);
		int maxThreadsPerBlock = properties.maxThreadsPerBlock;
		int threadsPerBlock = std::sqrt(maxThreadsPerBlock);

		//get number of levels. 
		int levels = (1 << 16) - 1;
		int* d_LUT;
		
		checkCudaErrors(cudaAlloc(d_LUT, levels));
		checkCudaErrors(cudaMemset(d_LUT, 0, sizeof(int)*levels));

		int lutBlocks = levels / maxThreadsPerBlock;

		//generate the look up table. 
		GpuTimer timer;
		timer.Start();
		generateLUT << <lutBlocks, maxThreadsPerBlock >> > (d_LUT, window, level, levels);
		timer.Stop();
		float msLut = timer.Elapsed();
#if PRINT_INFO
		printf("LUT calc used %d blocks\n", lutBlocks);
		printf("LUT kernel took %f ms\n", msLut);
#endif		
		//specify block size. 
		dim3 block_size(threadsPerBlock, threadsPerBlock);
		/*
		* Specify the grid size for the GPU.
		* Make it generalized, so that the size of grid changes according to the input image size
		*/
		dim3 grid_size;
		grid_size.x = (numCols + block_size.x - 1) / block_size.x;  /*< Greater than or equal to image width */
		grid_size.y = (numRows + block_size.y - 1) / block_size.y; /*< Greater than or equal to image height */

		//allocate device memory.
		input.gpuAlloc();
		out.gpuAlloc();
		//copy host memory to device. 
		input.memcpy(MemcpyDirection::HOST_TO_DEVICE);

		//initialize the timer. 
		GpuTimer winT;

		gimage::Type t = input.getType();
		switch (t) {
			case Type::UINT16:
				//now actually apply the window and leveling. 
				winT.Start();
				cudaWindowLevel << <grid_size, block_size >> >(static_cast<uint16_t*>(input.deviceData()), 
																static_cast<uint16_t*>(out.deviceData()), 
																d_LUT, numRows, numCols, window, level, levels);
				winT.Stop();
				break;
		}

		float ms = winT.Elapsed();
#if PRINT_INFO
		printf("WindowLevel kernel took %f ms\n", ms);
#endif
		//copy back data. 
		out.memcpy(MemcpyDirection::DEVICE_TO_HOST);

		//clean up. 
		input.gpuFree();
		out.gpuFree();
		checkCudaErrors(cudaFree(d_LUT));
	}

	/**
	* Performs canny edge detection on the input and outputs an image with only the edges in the output. 
	* This performs the following steps: (1) Apply a gaussian filter to smooth the image.
	* (2) Apply the Sobel operator in the x and y directions and keep track of gradient/direction.
	* (3) Perform non maximum suppression on the image.
	* (4) Use dual hysteresis thresholding to further eliminate false edges. 
	*/
	void GIMAGE_EXPORT cannyEdgeDetector(Array& input, Array& output, int numRows, int numCols,
		float sigma, int lowerThresh, int upperThresh) {

		assert(input.getType() == output.getType());
		assert(sigma > 0);
		assert(lowerThresh < upperThresh);
		assert(lowerThresh > 0 && upperThresh > 0);

		//set the device. 
		int device = selectDevice();
		checkCudaErrors(cudaSetDevice(device));
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);

		//calculate the threds per block. 
		int maxThreadsPerBlock = properties.maxThreadsPerBlock;
		int threadsPerBlock = std::sqrt(maxThreadsPerBlock);

		//specify block size. 
		dim3 block_size(threadsPerBlock, threadsPerBlock);
		/*
		* Specify the grid size for the GPU.
		* Make it generalized, so that the size of grid changes according to the input image size
		*/
		dim3 grid_size;
		grid_size.x = (numCols + block_size.x - 1) / block_size.x;  /*< Greater than or equal to image width */
		grid_size.y = (numRows + block_size.y - 1) / block_size.y; /*< Greater than or equal to image height */

	
		//create Sobel kernels
		int *k_gx = new int[9];
		int *k_gy = new int[9];
		k_gx[0] = -1; k_gx[1] = 0; k_gx[2] = 1;
		k_gx[3] = -2; k_gx[4] = 0; k_gx[5] = 2;
		k_gx[6] = -1; k_gx[7] = 0; k_gx[8] = 1;

		k_gy[0] = 1; k_gy[1] = 2; k_gy[2] = 1;
		k_gy[3] = 0; k_gy[4] = 0; k_gy[5] = 0;
		k_gy[6] = -1; k_gy[7] = -2; k_gy[8] = -1;

		//create device copies of the sobel kernels.
		int* d_kgx;
		int *d_kgy;
		checkCudaErrors(cudaAlloc(d_kgx, 9));
		checkCudaErrors(cudaAlloc(d_kgy, 9));
		checkCudaErrors(cudaMemcpy(d_kgx, k_gx, sizeof(int) * 9, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_kgy, k_gy, sizeof(int) * 9, cudaMemcpyHostToDevice));

		int *d_theta;
		checkCudaErrors(cudaAlloc(d_theta, input.size()));

		//check the image type.
		gimage::Type t = input.getType();
		switch (t) {
			case Type::UINT16:
				gimage::ArrayUint16 blurred(numRows, numCols);
				//run gaussian blur first. 
				gaussianBlur(input, blurred, sigma, numRows, numCols, 5);
				uint16_t *d_gradient;
				
				//allocate all our arrays. 
				checkCudaErrors(cudaAlloc(d_gradient, input.size()));
		
				//allocate on gpu. 
				input.gpuAlloc();
				//copy data to GPU.
				input.memcpy(MemcpyDirection::HOST_TO_DEVICE);

				uint16_t* d_in;
				uint16_t* d_out;
				//get input device pointer. 
				d_in = static_cast<uint16_t*>(input.deviceData());
				
				GpuTimer timer;
				timer.Start();
				//call our gradient kernel
				gradientAndDirection << <grid_size, block_size >> >(d_in, d_gradient, d_theta, d_kgx, d_kgy, numRows, numCols);
				timer.Stop();
				float gradMs = timer.Elapsed();
#if PRINT_INFO
				printf("Gradient kernel took %f ms\n", gradMs);
#endif
				//synchronize the device. 
				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

				//allocated output.
				output.gpuAlloc();
				d_out = static_cast<uint16_t*>(output.deviceData());

				//create non maximum suppression output.
				ArrayUint16 nonMaxOut(output.rows, output.cols);
				//allocate on GPU.
				nonMaxOut.gpuAlloc();
				//get device pointer.
				uint16_t* d_nMax_out = static_cast<uint16_t*>(nonMaxOut.deviceData());
				
				//set output to zeros.
				checkCudaErrors(cudaMemset(d_out, 0, output.totalSize()));
				timer.Start();
				//perform non maximum suppression. 
				nonMaximumSuppression << <grid_size, block_size >> >(d_gradient, d_theta, d_nMax_out, numRows, numCols);
				timer.Stop();
				float nonMaxMs = timer.Elapsed();
#if PRINT_INFO
				printf("Non-maximum kernel took %f ms\n", nonMaxMs);
#endif
				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

				timer.Start();
				hysteresisThresholding << <grid_size, block_size >> >(d_nMax_out, d_out, d_theta, upperThresh, lowerThresh, numRows, numCols);
				timer.Stop();
				float hysMs = timer.Elapsed();
#if PRINT_INFO
				printf("Thresholding kernel took %f ms\n", hysMs);
#endif
				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

				//copy result to gpu. 
				checkCudaErrors(cudaMemcpy(static_cast<uint16_t*>(output.hostData()), d_out, output.totalSize(), cudaMemcpyDeviceToHost));

				//free up used memory. 
				input.gpuFree();
				output.gpuFree();
				nonMaxOut.gpuFree();

				checkCudaErrors(cudaFree(d_gradient));
				checkCudaErrors(cudaFree(d_theta));
				break;
		}
		
		//free our gpu filters. 
		checkCudaErrors(cudaFree(d_kgx));
		checkCudaErrors(cudaFree(d_kgy));

		//free cpu memory. 
		delete[] k_gx;
		delete[] k_gy;
	}
}

