#ifndef GIMAGE_H
#define GIMAGE_H

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cassert>
#include <algorithm>
#include "gimage_export.h"
#include "timer.h"

#define PRINT_INFO 1
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

template<typename T>
__global__ void test(T* a1, T* a2, T* out) {
	static_assert(std::is_arithmetic<T>::value, "Only arithmetic types allowed.");
	int id = threadIdx.x;
	out[id] = a1[id] + a2[id];
}

namespace gimage {
	void GIMAGE_EXPORT gaussianBlur(uint16_t *input, uint16_t *output, float sigma, int numRows, int numCols, int blurSize);
	void GIMAGE_EXPORT windowAndLevel(uint16_t *input, uint16_t *out, int numRows, int numCols, int window, int level);
	void GIMAGE_EXPORT cannyEdgeDetector(uint16_t *input, uint16_t *output, int numRows, int numCols,
										float sigma, uint16_t lowerThresh, uint16_t upperThresh);
	template<typename T>
	void test(T* first, T* second, T* out, int size) {
		int threads = size;
		T* d_first;
		T* d_second;
		T* d_out;
		checkCudaErrors(cudaMalloc(&d_first, sizeof(T)*size));
		checkCudaErrors(cudaMalloc(&d_second, sizeof(T)*size));
		checkCudaErrors(cudaMalloc(&d_out, sizeof(T)*size));
		checkCudaErrors(cudaMemcpy(d_first, first, sizeof(T)*size, cudaMemcpyHostToDevice));
		test << <1, threads >> >(first, second, out);
	}
}

#endif //GIMAGE_H