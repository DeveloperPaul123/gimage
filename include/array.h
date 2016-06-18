#ifndef ARRAY_H
#define ARRAY_H

#include <cstdint>
#include <memory>
#include "gimage_export.h"

namespace gimage {

	enum class MemcpyDirection {
		HOST_TO_DEVICE,
		DEVICE_TO_HOST
	};

	enum class Type {
		NONE,
		DOUBLE,
		INT,
		FLOAT,
		UINT16, 
		UINT8, 
		LONG
	};

	/**
	* Base array class. Holds data type of underlying data as an enum. 
	*/
	class GIMAGE_EXPORT Array {
	public:

		/**
		* Base array class that holds image and other data. 
		* @param rows number of rows in the array.
		* @param cols the number of columns in the array. 
		*/
		Array(int rows, int cols, Type type) {
			this->rows = rows;
			this->cols = cols;
			_type = type;
			_size = this->rows*this->cols;
		}

		/**
		* Virtual deconstructor. All derived classes should clean up
		* their data here. 
		*/
		virtual ~Array(){}

		/**
		* Subtraction operator, subtracts arrays itemwise.
		* Arrays must be the same size.
		*/
		virtual Array& operator-(Array &other) = 0;

		/**
		* Addition operator, adds arrays item wise. Arrays
		* must be the same size.
		*/
		virtual Array& operator+(Array &other) = 0;

		/**
		* Assignment operator. Used to copy data from one array
		* to another. 
		*/
		virtual Array& operator=(Array &other) = 0;

		/**
		* Returns the array's type.
		* @return Type the array's type.
		*/
		Type getType() {
			return _type;
		}

		/**
		* Get the underlying data of the array. 
		* @return void* raw data pointer. 
		*/
		virtual void* hostData() = 0;

		/**
		* Get the underlying data of the gpu.
		* @return void* raw device data pointer. 
		*/
		virtual void* deviceData() = 0;

		/**
		* Allocate data on the GPU. 
		* @return void* pointer to the gpu array.
		*/
		virtual void gpuAlloc() = 0;

		/**
		* Free data on the gpu. 
		*/
		virtual void gpuFree() = 0;

		/**
		* Clones data to another array.
		* @param other the other array to copy data to. 
		*/
		virtual void clone(Array& other) = 0;

		/**
		* Copy data to or from the device (GPU) to or from the host.
		* @param dir the direction to copy from. 
		*/
		virtual void memcpy(MemcpyDirection dir) = 0;

		/**
		* Total size of the array (i.e. size * sizeof(type))
		* @return size_t total size of the array.
		*/
		virtual int totalSize() = 0;

		/**
		* Set data value of array at a give row and column. 
		* @param row the row to insert at. 
		* @param col the column to insert at. 
		* @param value the value to insert. 
		*/
		template<typename T>
		void setData(int row, int col, T value) {
			assert(row < rows && col < cols);
			static_cast<T*>(hostData())[row*cols + col] = static_cast<T>(value);
		}

		/**
		* Get a value at a given row and column. 
		* @param row the row to look at.
		* @param col the column to look at. 
		*/
		template<typename T>
		T at(int row, int col) {
			assert(row < rows && col < cols);
			return static_cast<T*>(hostData())[row*cols + col];
		}

		/**
		* Get the total number of elements in the array.
		* @return int the total number of elements in the array. 
		*/
		int size() {
			return _size;
		}

		/**
		* Set the size of the array.
		*/
		void setSize(int size) {
			_size = size;
		}

		int rows = 0;
		int cols = 0;
	private:
		/**
		* Allocate data array on the host. 
		* @param size the size of the array to allocate. 
		*/
		virtual void allocate(int size) = 0;
		Type _type = Type::NONE;
		
		int _size = 0;
	};

	class GIMAGE_EXPORT DoubleArray : public Array {

	public:
		/**
		* Generic array of doubles.
		* @param rows number of rows in the array.
		* @param cols number of columns in the array.
		*/
		DoubleArray(int rows, int cols);

		/**
		* Copy constructor. This will copy all the data in other to this
		* double array.
		* @param other the array to copy.
		*/
		DoubleArray(DoubleArray &other);

		/**
		* Deconstructor. All data, including on the GPU
		* is deallocated here if it already hasn't been freed.
		*/
		~DoubleArray();

		Array& operator+(Array &other);

		Array& operator-(Array &other);

		Array& operator=(Array &other);

		/**
		* Returns a pointer to the host data.
		* @return void* host data pointer. Can static_cast this to double*
		*/
		void* hostData();

		/**
		* Returns a pointer to the device data array. Note that this array will be NULL
		* if gpuAlloc() has not been called.
		* @return void* device data pointer. Can be static_cast to double*
		*/
		/**
		* Returns a pointer to the device data array. Note that this array will be NULL
		* if gpuAlloc() has not been called.
		* @return void* device data pointer. Can be static_cast to double*
		*/
		void* deviceData();

		/**
		* Allocate data onto the GPU. Note that this does not copy data over to the GPU.
		*/
		void gpuAlloc();

		/**
		* Free GPU data. This function will check to see if the data pointer is
		* valid first before attempting to free it. It will be set to NULL once
		* it is freed from GPU memory.
		*/
		void gpuFree();

		/**
		* Clones host data from this array to the other array.
		* @param other the array to copy data to.
		*/
		void clone(Array& other);

		/**
		* Copy data to or from the host and/or device.
		* @param dir the direction to copy.
		*/
		void memcpy(MemcpyDirection dir);

		/**
		* Total size of the array including the size of the type.
		* @return the size of the array * sizeof(type)
		*/
		int totalSize();

	private:

		/**
		* Allocate host memory.
		* @param size the size of the data to allocate.
		*/
		virtual void allocate(int size);
		double *h_data;
		double *d_data = NULL;
	};

	class GIMAGE_EXPORT ArrayUint16 : public Array {

	public:
		/**
		* Generic array of unsigned 16 bit integers (uint16_t).
		* @param rows number of rows in the array.
		* @param cols number of columns in the array.
		*/
		ArrayUint16(int rows, int cols);

		/**
		* Copy constructor. This will copy all the data in other to this
		* array.
		* @param other the array to copy.
		*/
		ArrayUint16(ArrayUint16 &other);

		/**
		* Deallocate the array and underlying buffers.
		*/
		~ArrayUint16();

		Array& operator+(Array &other);

		Array& operator-(Array &other);

		Array& operator=(Array &other);

		/**
		* Returns a pointer to the host data.
		* @return void* host data pointer. Can static_cast this to uint16_t*
		*/
		void* hostData();

		/**
		* Returns a pointer to the device data array. Note that this array will be NULL
		* if gpuAlloc() has not been called.
		* @return void* device data pointer. Can be static_cast to uint16_t*
		*/
		void* deviceData();

		/**
		* Allocate data onto the GPU. Note that this does not copy data over to the GPU.
		* @return void* device pointer to data. Use static cast to cast this to the proper type.
		*/
		void gpuAlloc();

		/**
		* Free GPU data. This function will check to see if the data pointer is
		* valid first before attempting to free it. It will be set to NULL once
		* it is freed from GPU memory.
		*/
		void gpuFree();

		/**
		* Clones host data from this array to the other array.
		* @param other the array to copy data to.
		*/
		void clone(Array& other);

		/**
		* Copy data to or from the host and/or device.
		* @param dir the direction to copy.
		*/
		void memcpy(MemcpyDirection dir);

		/**
		* Total size of the array.
		* @return int the size of the array * sizeof(type)
		*/
		int totalSize();

	private:

		/**
		* Allocate host memory.
		* @param size the size of the data to allocate.
		*/
		virtual void allocate(int size);
		uint16_t *h_data;
		uint16_t *d_data = NULL;
	};

	class GIMAGE_EXPORT ArrayUint8 : public Array {

	public:
		/**
		* Generic array of unsigned 8 bit integers (uint16_t).
		* @param rows number of rows in the array.
		* @param cols number of columns in the array.
		*/
		ArrayUint8(int rows, int cols);

		/**
		* Copy constructor. This will copy all the data in other to this
		* array.
		* @param other the array to copy.
		*/
		ArrayUint8(ArrayUint8 &other);

		/**
		* Deallocate the array and underlying buffers.
		*/
		~ArrayUint8();

		Array& operator+(Array &other);

		Array& operator-(Array &other);

		Array& operator=(Array &other);

		/**
		* Returns a pointer to the host data.
		* @return void* host data pointer. Can static_cast this to uint8_t*
		*/
		void* hostData();

		/**
		* Returns a pointer to the device data array. Note that this array will be NULL
		* if gpuAlloc() has not been called.
		* @return void* device data pointer. Can be static_cast to uint8_t*
		*/
		void* deviceData();

		/**
		* Allocate data onto the GPU. Note that this does not copy data over to the GPU.
		* @return void* device pointer to data. Use static cast to cast this to the proper type.
		*/
		void gpuAlloc();

		/**
		* Free GPU data. This function will check to see if the data pointer is
		* valid first before attempting to free it. It will be set to NULL once
		* it is freed from GPU memory.
		*/
		void gpuFree();

		/**
		* Clones host data from this array to the other array.
		* @param other the array to copy data to.
		*/
		void clone(Array& other);

		/**
		* Copy data to or from the host and/or device.
		* @param dir the direction to copy.
		*/
		void memcpy(MemcpyDirection dir);

		/**
		* Total size of the array.
		* @return int the size of the array * sizeof(type)
		*/
		int totalSize();

	private:

		/**
		* Allocate host memory.
		* @param size the size of the data to allocate.
		*/
		virtual void allocate(int size);
		uint8_t *h_data;
		uint8_t *d_data = NULL;
	};

	class Image : public ArrayUint8 {

	};

	class Image16 : public ArrayUint16 {

	};

	class GIMAGE_EXPORT MatrixD :  public DoubleArray {
	public:
		/**
		* Matrix of doubles. Allocate a vector of a give size.
		* @param items the number of items in the matrix.
		*/
		MatrixD(int items);

		/**
		* Matrix of double. Allocate a two dimensional matrix.
		* @param rows the number of rows in the matrix.
		* @param cols the number of columns in the matrix.
		*/
		MatrixD(int rows, int cols);

		/**
		* Caculates the determinant of the matrix. This function asserts that
		* the matrix is square, i.e. the rows() == cols().
		* @return double the determinant of the array. 
		*/
		double det();

		/**
		* Multiplication operator for a matrix. This assumes that
		* the inner dimensions of the two arrays match. For example if 
		* this array is a mx3 array then other must be a 3xn array. The returned array
		* will be an mxn array.
		* @return Array the resultant array. 
		*/
		MatrixD operator*(MatrixD other);

	};
}
#endif