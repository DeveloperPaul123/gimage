#ifndef ARRAY_H
#define ARRAY_H

#include <cstdint>
#include "gimage_export.h"

namespace gimage {
	template<typename T>
	class GIMAGE_EXPORT InputArray {
	public:
		InputArray() {
			_size = 0;
		}

		InputArray(size_t size) {
			_size = size;
			allocate(size);
		}

		~InputArray() {
			delete[] data;
		}

		T* getData() {
			return data;
		}

		T get(int index) {
			return data[index];
		}

		void setData(T elem, int index) {
			data[index] = elem;
		}

		size_t totalSize() {
			return sizeof(T)*_size;
		}

		size_t size() {
			return _size;
		}
	private:
		void allocate(size_t size) {
			data = new T[size];
		}
		size_t _size;
		T* data;
	};

	enum Type {
		TYPE_UCHAR = 0,
		TYPE_UINT16 = 1
	};

	class GIMAGE_EXPORT Matrix {
	public:
		Matrix(Type type, size_t size);
		Type type();
		size_t size();
		template<typename T>
		T* data() {
			return nullptr;
		}

	private:
		size_t _size;
		Type _type;
	};

	class GIMAGE_EXPORT MatrixUint16 : public Matrix, public InputArray < uint16_t > {
	public:
		MatrixUint16(size_t size);
		uint16_t* data();
	};

	class GIMAGE_EXPORT MatrixUint8 : public Matrix, public InputArray < uint8_t > {
	public:
		MatrixUint8(size_t size);
	};
}
#endif