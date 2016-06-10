#include "gimage.h"
#include "array.h"
#include "timer.h"

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
	int r = threadIdx.x + blockIdx.x*blockDim.x;
	int c = threadIdx.y + blockIdx.y*blockDim.y;
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
	int r = threadIdx.x + blockIdx.x*blockDim.x;
	int c = threadIdx.y + blockIdx.y*blockDim.y;
	//get unique point in image by finding position in grid.
	int index = r + c*blockDim.x*gridDim.x;
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
	int r = threadIdx.x + blockIdx.x*blockDim.x;
	int c = threadIdx.y + blockIdx.y*blockDim.y;
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
__global__ void nonMaximumSuppression(T* d_gradMag, int* d_theta, T* d_out, int upperThresh, int lowerThresh, int numRows, int numCols) {
	//get row and column in the current grid (this should be a sub set of the image if it is large enough.
	int r = threadIdx.x + blockIdx.x*blockDim.x;
	int c = threadIdx.y + blockIdx.y*blockDim.y;
	//get unique point in image by finding position in grid.
	int index = c + r*blockDim.x*gridDim.x;

	if (index >= numRows*numCols) {
		return;
	}

	//TODO: Perform suppression. 
}

/**
* Namespace for all gimage functions.
*/
namespace gimage {

	/**
	* Return the type of the array. See gimage::Type for possible types. 
	* @return Type the type of the array.
	*/
	Type Array::getType() {
		return _type;
	}

	/**
	* Generic array of doubles.
	* @param rows number of rows in the array.
	* @param cols number of columns in the array. 
	*/
	DoubleArray::DoubleArray(int rows, int cols) : Array(rows, cols, Type::DOUBLE) {
		allocate(size());
	}

	/**
	* Deallocate all data. 
	*/
	DoubleArray::~DoubleArray() {
		delete[] h_data;
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}
	}

	/**
	* Returns a pointer to the host data. 
	* @return void* host data pointer. Can static_cast this to double*
	*/
	void* DoubleArray::hostData() {
		return h_data;
	}

	/**
	* Returns a pointer to the device data array. Note that this array will be NULL
	* if gpuAlloc() has not been called. 
	* @return void* device data pointer. Can be static_cast to double*
	*/
	void* DoubleArray::deviceData() {
		return d_data;
	}

	/**
	* Allocate data onto the GPU. Note that this does not copy data over to the GPU.
	*/
	void DoubleArray::gpuAlloc() {
		if (!d_data) {
			checkCudaErrors(cudaAlloc(d_data, size()));
		}
	}

	/**
	* Free GPU data. This function will check to see if the data pointer is
	* valid first before attempting to free it. It will be set to NULL once 
	* it is freed from GPU memory.
	*/
	void DoubleArray::gpuFree() {
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}	
	}

	/**
	* Total size of the array including the size of the type.
	* @return the size of the array * sizeof(type)
	*/
	int DoubleArray::totalSize() {
		return size() * sizeof(double);
	}

	/**
	* Clones host data from this array to the other array.
	* @param other the array to copy data to. 
	*/
	void DoubleArray::clone(Array& other) {
		assert(other.getType() == getType());
		assert(other.size() == size());
		std::memcpy(static_cast<double*>(other.hostData()), 
			static_cast<double*>(hostData()), totalSize());
	}

	/**
	* Copy data to or from the host and/or device. 
	* @param dir the direction to copy. 
	*/
	void DoubleArray::memcpy(MemcpyDirection dir) {
		if (dir == MemcpyDirection::HOST_TO_DEVICE) {
			checkCudaErrors(cudaMemcpy(d_data, h_data, totalSize(), cudaMemcpyHostToDevice));
		}
		else {
			checkCudaErrors(cudaMemcpy(h_data, d_data, totalSize(), cudaMemcpyDeviceToHost));
		}
	}

	/**
	* Allocate host memory. 
	* @param size the size of the data to allocate. 
	*/
	void DoubleArray::allocate(int size) {
		h_data = new double[size];
	}

	/**
	* Generic array of unsigned 16 bit integers.
	* @param rows number of rows in the array.
	* @param cols number of columns in the array.
	*/
	ArrayUint16::ArrayUint16(int rows, int cols) : Array(rows, cols, Type::UINT16) {
		allocate(size());
	}

	/**
	* Deallocate the array and underlying buffers. 
	*/
	ArrayUint16::~ArrayUint16() {
		delete[] h_data;
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}
	}

	/**
	* Returns a pointer to the host data.
	* @return void* host data pointer. Can static_cast this to double*
	*/
	void* ArrayUint16::hostData() {
		return h_data;
	}

	/**
	* Returns a pointer to the device data array. Note that this array will be NULL
	* if gpuAlloc() has not been called.
	* @return void* device data pointer. Can be static_cast to double*
	*/
	void* ArrayUint16::deviceData() {
		return d_data;
	}

	/**
	* Allocate data onto the GPU. Note that this does not copy data over to the GPU.
	* @return void* device pointer to data. Use static cast to cast this to the proper type. 
	*/
	void ArrayUint16::gpuAlloc() {
		if (!d_data) {
			checkCudaErrors(cudaAlloc(d_data, size()));
		}
	}

	/**
	* Free GPU data. This function will check to see if the data pointer is
	* valid first before attempting to free it. It will be set to NULL once
	* it is freed from GPU memory.
	*/
	void ArrayUint16::gpuFree() {
		if (d_data) {
			checkCudaErrors(cudaFree(d_data));
			d_data = NULL;
		}	
	}

	/**
	* Total size of the array.
	* @return int the size of the array * sizeof(type)
	*/
	int ArrayUint16::totalSize() {
		return size() * sizeof(uint16_t);
	}

	/**
	* Copy data to or from the host and/or device.
	* @param dir the direction to copy.
	*/
	void ArrayUint16::memcpy(MemcpyDirection dir) {
		if (dir == MemcpyDirection::HOST_TO_DEVICE) {
			checkCudaErrors(cudaMemcpy(d_data, h_data, totalSize(), cudaMemcpyHostToDevice));
		}
		else {
			checkCudaErrors(cudaMemcpy(h_data, d_data, totalSize(), cudaMemcpyDeviceToHost));
		}
	}

	/**
	* Clones host data from this array to the other array.
	* @param other the array to copy data to.
	*/
	void ArrayUint16::clone(Array& other) {
		assert(other.getType() == getType());
		assert(other.size() == size());
		std::memcpy(static_cast<uint16_t*>(other.hostData()), 
			static_cast<uint16_t*>(hostData()), totalSize());
	}

	/**
	* Allocate host memory.
	* @param size the size of the data to allocate.
	*/
	void ArrayUint16::allocate(int size) {
		h_data = new uint16_t[size];
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
		assert(input.rows() == out.rows() && input.cols() == out.cols() && input.rows() == numRows && input.cols() == numCols);
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
	* @param input the input image. 
	* @param output the output image (edges only).
	* @param numRows number of rows in the image.
	* @param numCols the number of the columns in the image.
	* @param sigma sigma for the gaussian blur.
	* @param uint16_t lowerThresh lower threshold for the canny edge detector.
	* @param uint16_t upterThresh upper threshold for the canny edge detector. 
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

		//check the image type.
		gimage::Type t = input.getType();
		switch (t) {
			case Type::UINT16:
				gimage::ArrayUint16 blurred(numRows, numCols);
				//run gaussian blur first. 
				gaussianBlur(input, blurred, sigma, numRows, numCols, 5);
				uint16_t *d_gradient;
				int *d_theta;
				//allocate all our arrays. 
				checkCudaErrors(cudaAlloc(d_gradient, input.size()));
				checkCudaErrors(cudaAlloc(d_theta, input.size()));
			
				//allocate on gpu. 
				input.gpuAlloc();
				input.memcpy(MemcpyDirection::HOST_TO_DEVICE);

				uint16_t* d_in;
				uint16_t* d_out;
				d_in = static_cast<uint16_t*>(input.deviceData());
				
				//call our gradient kernel
				gradientAndDirection << <grid_size, block_size >> >(d_in, d_gradient, d_theta, d_kgx, d_kgy, numRows, numCols);
				//synchronize the device. 
				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

				//allocated output.
				output.gpuAlloc();
				d_out = static_cast<uint16_t*>(output.deviceData());

				nonMaximumSuppression << <grid_size, block_size >> >(d_gradient, d_theta, d_out, upperThresh, lowerThresh, numRows, numCols);
				//for now just testing gradient part so copy this result to the output. 
				checkCudaErrors(cudaMemcpy(static_cast<uint16_t*>(output.hostData()), d_gradient, output.totalSize(), cudaMemcpyDeviceToHost));

				//free up used memory. 
				input.gpuFree();
				output.gpuFree();

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

