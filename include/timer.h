#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

class GpuTimer
{
private:
  cudaEvent_t start;
  cudaEvent_t stop;

public:
	/**
	* Helper class to time GPU kernels. This class binds to cuda events 
	* to accurately measure the time it takes to execute a kernel.
	*/
	GpuTimer()
	{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	}

	/**
	* Start the timer.
	*/
	void Start()
	{
	cudaEventRecord(start, 0);
	}

	/**
	* Stop the timer. 
	*/
	void Stop()
	{
	cudaEventRecord(stop, 0);
	}

	/**
	* Get the elapsed time in milliseconds.
	* @return float the elapsed time in miliseconds. 
	*/
	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

#endif  /* GPU_TIMER_H__ */