#ifndef ARRAY_H
#define ARRAY_H

#include <cstdint>
#include <memory>
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
}
#endif