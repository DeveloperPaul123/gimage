#ifndef ARRAY_H
#define ARRAY_H

#include <cstdint>
#include <memory>
#include "gimage_export.h"

namespace gimage {

	enum Type {
		TYPE_NONE = 0,
		TYPE_DOUBLE = 1,
		TYPE_INT = 2,
		TYPE_FLOAT = 3,
		TYPE_UINT16 = 4, 
		TYPE_UINT8 = 5, 
		TYPE_LONG = 6
	};

	/**
	* Base array class. Holds data type of underlying data as an enum. 
	*/
	class GIMAGE_EXPORT Array {
	public:
		/**
		* Array of a given Type. Type can be NONE (default),
		* double, int, float, uint16_t and uint8_t
		*/
		Array(Type type);
		/**
		* Returns the array's type.
		* @return Type the array's type.
		*/
		Type getType();

		/**
		* Get the underlying data of the array. 
		* @return void* raw data pointer. 
		*/
		virtual void* data() = 0;

		/**
		* Total size of the array (i.e. size * sizeof(type))
		* @return size_t total size of the array.
		*/
		virtual size_t totalSize() = 0;

		/**
		* Number of elements in the array. 
		* @return size_t total size. 
		*/
		virtual size_t size() = 0;

		/**
		* Get the number of rows in the array.
		* @return int the number of rows.
		*/
		virtual int rows() = 0;

		/**
		* Get the number of columns in the array.
		* @return int the number of columns in the array.
		*/
		virtual int cols() = 0;

	private:
		Type _type;
	};

	/**
	* Template Matrix class for different data types. 
	*/
	template<typename T>
	class GIMAGE_EXPORT Matrix : public Array {
	public:

		Matrix(int rows, int columns) : Array(TYPE_NONE){
			_rows = rows; 
			_cols = columns;
			_size = _rows*_cols;
			allocate(_size);
		}

		/**
		* Instantiate matrix of a given size. 
		* This will allocate memory of type T[size].
		* @param size the size of the matrix. 
		*/
		Matrix(size_t size) : Array(TYPE_NONE) {
			_size = size;
			allocate(size);
		}

		/**
		* Deconstructor. Deletes the previously allocated memory. 
		*/
		~Matrix() {
			delete[] _data;
		}

		/**
		* Get data at a particular row and column. 
		* @param row the row to look at.
		* @param col the column to look at. 
		*/
		T at(int row, int col) {
			assert(row < _rows && col < _cols);
			return _data[row*_cols + col];
		}

		/**
		* Returns a pointer to the underlying data. 
		* @return void* data pointer. 
		*/
		virtual void* data() {
			return static_cast<T*>(getData());
		}

		virtual int rows() {
			return _rows;
		}

		virtual int cols() {
			return _cols;
		}
		/**
		* Gets data from a given index. 
		*/
		T get(int index) {
			return _data[index];
		}

		/**
		* Set data at a given index. 
		*/
		void setData(T elem, int index) {
			_data[index] = elem;
		}

		/**
		* Returns the total size of this matrix. This includes the size of
		* the type and the size of the array.
		* @return size_t total size. 
		*/
		virtual size_t totalSize() {
			return sizeof(T)*_size;
		}

		/**
		* Returns the size of the array.
		* @return size_t the size. 
		*/
		virtual size_t size() {
			return _size;
		}
	private:
		/**
		* Private get data member for getting the internal data. 
		* @return T* pointer to internal data. 
		*/
		T* getData() {
			return _data;
		}
		
		/**
		* Allocate an array of a given size.
		* @param size_t size the size.
		*/
		void allocate(size_t size) {
			_data = new T[size];
		}
		
		int _cols = 0;
		int _rows = 0;
		size_t _size;
		T* _data;
	};

	typedef Matrix < double > MatrixD;
	typedef Matrix<int> MatrixI;
	typedef Matrix<float> MatrixF;
	typedef Matrix<long> MatrixL;
	typedef Matrix<uint16_t> MatrixU16;
	typedef Matrix<uint8_t> MatrixU8;
}
#endif