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

namespace gimage {
	void GIMAGE_EXPORT gaussianBlur(uint16_t *input, uint16_t *output, int numRows, int numCols, int blurSize);
	void GIMAGE_EXPORT windowAndLevel(uint16_t *input, uint16_t *out, int numRows, int numCols, int window, int level);
}

#endif //GIMAGE_H