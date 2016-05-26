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
		virtual void* data() = 0;
		virtual size_t totalSize() = 0;
		virtual size_t size() = 0;

	private:
		Type _type;
	};

	/**
	* Template Matrix class for different data types. 
	*/
	template<typename T>
	class GIMAGE_EXPORT Matrix : public Array {
	public:
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
		* Returns a pointer to the underlying data. 
		* @return void* data pointer. 
		*/
		virtual void* data() {
			return static_cast<T*>(getData());
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