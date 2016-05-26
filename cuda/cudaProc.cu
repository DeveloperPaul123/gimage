#include "gimage.h"
#include "array.h"
#include "timer.h"

#define PI 3.14159265359
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

__global__ void cannyEdge(uint16_t *d_in, uint16_t *d_out, double* d_theta, int* k_gx, int* k_gy, int numRows, int numCols) {
	int r = threadIdx.x + blockIdx.x*blockDim.x;
	int c = threadIdx.y + blockIdx.y*blockDim.y;
	//get unique point in image by finding position in grid.
	int index = r + c*blockDim.x*gridDim.x;
	extern __shared__ uint16_t intermediate[];

	if (index > numRows*numCols) {
		return;
	}

	int x_res, y_res;
	int kernelSize = 9;
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

			int image_value = static_cast<int>(d_in[image_r * numCols + image_c]);
			int filter_x = k_gx[(filter_r + kernelSize / 2) * kernelSize + filter_c + kernelSize / 2];
			int filter_y = k_gx[(filter_r + kernelSize / 2) * kernelSize + filter_c + kernelSize / 2];
			//add filter value to result.
			x_res += image_value*filter_x;
			y_res += image_value*filter_y;
		}
	}

	intermediate[index] = x_res + y_res;
	double angle = atan2f(y_res, x_res) * (180.0 / PI);
	double correctAngle = round(angle / 45.0) * 45.0;
	d_theta[index] = correctAngle;

	//need all threads to be done before proceeding. 
	__syncthreads();

	double a = d_theta[index];
	//now need to compare values that are in the same direction. 

}

/**
* Namespace for all gimage functions.
*/
namespace gimage {
	/**
	* Performs a Gaussian blur on a given image and stores it in the output.
	* @param input the input image
	* @param output the output image.
	* @param numRows the number of rows in the input image.
	* @param numCols the number of columns int he input image.
	* @param blurSize the size of the blur. This must be odd. Note that the blur filter will be square.
	*/
	void GIMAGE_EXPORT gaussianBlur(uint16_t *input, uint16_t *output, float sigma, int numRows, int numCols, int blurSize) {
		if (blurSize % 2 == 0) {
			throw(std::exception("Blur size must be odd."));
		}
		else {

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
			int size = numRows*numCols*sizeof(uint16_t);
			uint16_t *d_in;
			uint16_t *d_out;
			float *d_filter;

			//allocated memory for filter. 
			checkCudaErrors(cudaMalloc(&d_filter, sizeof(float)*blurSize*blurSize));
			checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*blurSize*blurSize, cudaMemcpyHostToDevice));

			//allocate image memory.
			checkCudaErrors(cudaMalloc(&d_in, size));
			checkCudaErrors(cudaMalloc(&d_out, size));
			checkCudaErrors(cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice));

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
			GpuTimer t;
			t.Start();
			//call the kernal.
			gaussian << <grid_size, block_size >> >(d_in, d_out, d_filter, numRows, numCols, blurSize);
			t.Stop();
			float ms = t.Elapsed();
#if PRINT_INFO
			printf("Kernel took %f ms\n", ms);
#endif
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			checkCudaErrors(cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost));

			//clean up
			checkCudaErrors(cudaFree(d_in));
			checkCudaErrors(cudaFree(d_out));
			checkCudaErrors(cudaFree(d_filter));

			free(h_filter);
		}
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
		
		//assert the same type.
		assert(input.getType() == out.getType());
		//assert same size. 
		assert(input.size() == out.size());
		//select the device
		int device = selectDevice();
		checkCudaErrors(cudaSetDevice(device));
		struct cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);
		int maxThreadsPerBlock = properties.maxThreadsPerBlock;
		int threadsPerBlock = std::sqrt(maxThreadsPerBlock);

		int levels = (1 << 16) - 1;
		int* d_LUT;
		checkCudaErrors(cudaMalloc(&d_LUT, sizeof(int)*levels));
		checkCudaErrors(cudaMemset(d_LUT, 0, sizeof(int)*levels));

		int lutBlocks = levels / maxThreadsPerBlock;

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

		//now actually apply the window and leveling. 
		gimage::Type t = input.getType();
		switch (t) {
		case TYPE_UINT16:
			//now we can filter the image. 
			int size = input.totalSize();
			uint16_t *d_in;
			uint16_t *d_out;

			//allocate image memory.
			checkCudaErrors(cudaMalloc(&d_in, size));
			checkCudaErrors(cudaMalloc(&d_out, size));
			checkCudaErrors(cudaMemcpy(d_in, static_cast<uint16_t*>(input.data()), size, cudaMemcpyHostToDevice));

			GpuTimer winT;
			winT.Start();
			cudaWindowLevel << <grid_size, block_size >> >(d_in, d_out, d_LUT, numRows, numCols, window, level, levels);
			winT.Stop();
			float ms = winT.Elapsed();
#if PRINT_INFO
			printf("WindowLevel kernel took %f ms\n", ms);
#endif
			//copy back data. 
			checkCudaErrors(cudaMemcpy(static_cast<uint16_t*>(out.data()), d_out, size, cudaMemcpyDeviceToHost));

			//clean up. 
			checkCudaErrors(cudaFree(d_in));
			checkCudaErrors(cudaFree(d_out));
			checkCudaErrors(cudaFree(d_LUT));
			break;
		}
	

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
	void GIMAGE_EXPORT cannyEdgeDetector(uint16_t *input, uint16_t *output, int numRows, int numCols,
		float sigma, uint16_t lowerThresh, uint16_t upperThresh) {

		//set the device. 
		int device = selectDevice();
		checkCudaErrors(cudaSetDevice(device));
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);

		int maxThreadsPerBlock = properties.maxThreadsPerBlock;
		int threadsPerBlock = std::sqrt(maxThreadsPerBlock);

		//run gaussian blur first. 
		gaussianBlur(input, output, sigma, numRows, numCols, 5);

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
		checkCudaErrors(cudaMalloc(&d_kgx, sizeof(int) * 9));
		checkCudaErrors(cudaMalloc(&d_kgy, sizeof(int) * 9));
		checkCudaErrors(cudaMemcpy(d_kgx, k_gx, sizeof(int) * 9, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_kgy, k_gy, sizeof(int) * 9, cudaMemcpyHostToDevice));


		//allocate all our arrays. 
		double *d_gx;
		double *d_gy;
		double *d_theta;
		checkCudaErrors(cudaMalloc(&d_gx, sizeof(double)*numRows*numCols));
		checkCudaErrors(cudaMalloc(&d_gy, sizeof(double)*numRows*numCols));
		checkCudaErrors(cudaMalloc(&d_theta, sizeof(double)*numRows*numCols));

		//free up used memory. 
		checkCudaErrors(cudaFree(d_gx));
		checkCudaErrors(cudaFree(d_gy));
		checkCudaErrors(cudaFree(d_theta));
		checkCudaErrors(cudaFree(d_kgx));
		checkCudaErrors(cudaFree(d_kgy));

		//free cpu memory too. 
		free(k_gx);
		free(k_gy);

	}
}

