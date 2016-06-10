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
			_rows = rows;
			_cols = cols;
			_type = type;
			_size = _rows*_cols;
		}

		virtual ~Array(){}

		/**
		* Returns the array's type.
		* @return Type the array's type.
		*/
		Type getType();

		/**
		* Get the underlying data of the array. 
		* @return void* raw data pointer. 
		*/
		virtual void* hostData() = 0;
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

		template<typename T>
		void setData(int row, int col, T value) {
			assert(row < _rows && col < _cols);
			static_cast<T*>(hostData())[row*_cols + col] = static_cast<T>(value);
		}

		template<typename T>
		T at(int row, int col) {
			assert(row < _rows && col < _cols);
			return static_cast<T*>(hostData())[row*_cols + col];
		}

		/**
		* Get the number of rows in the array.
		* @return int the number of rows.
		*/
		int rows() {
			return _rows;
		}

		/**
		* Get the number of columns in the array.
		* @return int the number of columns in the array.
		*/
		int cols() {
			return _cols;
		}

		int size() {
			return _size;
		}

	private:
		/**
		* Allocate data array on the host. 
		* @param size the size of the array to allocate. 
		*/
		virtual void allocate(int size) = 0;
		Type _type = Type::NONE;
		int _rows = 0;
		int _cols = 0;
		int _size = 0;
	};

	class GIMAGE_EXPORT DoubleArray : public Array {

	public:
		DoubleArray(int rows, int cols);
		~DoubleArray();
		void* hostData();
		void* deviceData();
		void gpuAlloc();
		void gpuFree();
		void clone(Array& other);
		void memcpy(MemcpyDirection dir);
		int totalSize();

	private:
		virtual void allocate(int size);
		double *h_data;
		double *d_data = NULL;
	};

	class GIMAGE_EXPORT ArrayUint16 : public Array {

	public:
		ArrayUint16(int rows, int cols);
		~ArrayUint16();
		void* hostData();
		void* deviceData();
		void gpuAlloc();
		void gpuFree();
		void clone(Array& other);
		void memcpy(MemcpyDirection dir);
		int totalSize();

	private:
		virtual void allocate(int size);
		uint16_t *h_data;
		uint16_t *d_data = NULL;
	};
}
#endif