#include "array.h"
#include <cstdint>
#include <memory>

namespace gimage {

	InputArray<uint16_t>::InputArray(size_t size) {
		_size = size;
		allocate(size);
	}

	void InputArray<uint16_t>::setData(uint16_t elem, int index) {
		data[index] = index;
	}

	uint16_t InputArray<uint16_t>::get(int index) {
		return data[index];
	}

	InputArray<uint8_t>::InputArray(size_t size) {
		_size = size;
		allocate(size);
	}

	void InputArray<uint8_t>::setData(uint8_t elem, int index) {
		data[index] = index;
	}

	uint8_t InputArray<uint8_t>::get(int index) {
		return data[index];
	}

	Matrix::Matrix(Type type, size_t size) {
		_size = size;
		_type = type;
	}

	Type Matrix::type() {
		return _type;
	}

	size_t Matrix::size() {
		return _size;
	}

	MatrixUint16::MatrixUint16(size_t size) : Matrix(TYPE_UINT16, size) , InputArray<uint16_t>(size){
	}

	uint16_t* MatrixUint16::data() {
		return getData();
	}

	MatrixUint8::MatrixUint8(size_t size) : Matrix(TYPE_UCHAR, size), InputArray<uint8_t>(size) {

	}
	
}
