#include "array.h"
#include "array.h"

namespace gimage {

	

	Matrix<double>::Matrix<double>(size_t size) : Array(TYPE_DOUBLE){
		_size = size;
		allocate(_size);
	}

	void* Matrix<double>::data() {
		return static_cast<double*>(getData());
	}
	void Matrix<double>::setData(double d, int index) {
		_data[index] = d;
	}
	Matrix<double>::~Matrix() {
		delete[] _data;
	}
	double Matrix<double>::get(int index) {
		return _data[index];
	}
	size_t Matrix<double>::totalSize() {
		return sizeof(double)*_size;
	}
	size_t Matrix<double>::size() {
		return _size;
	}


	Matrix<int>::Matrix<int>(size_t size) : Array(TYPE_INT){
		_size = size;
		allocate(_size);
	}
	void* Matrix<int>::data() {
		return static_cast<int*>(getData());
	}
	void Matrix<int>::setData(int d, int index) {
		_data[index] = d;
	}
	Matrix<int>::~Matrix() {
		delete[] _data;
	}
	int Matrix<int>::get(int index) {
		return _data[index];
	}
	size_t Matrix<int>::totalSize() {
		return sizeof(int)*_size;
	}
	size_t Matrix<int>::size() {
		return _size;
	}

	MatrixF::Matrix<float>(size_t size) : Array(TYPE_FLOAT){
		_size = size;
		allocate(_size);
	}
	void* MatrixF::data() {
		return static_cast<float*>(getData());
	}
	void MatrixF::setData(float d, int index) {
		_data[index] = d;
	}
	MatrixF::~Matrix() {
		delete[] _data;
	}
	float Matrix<float>::get(int index) {
		return _data[index];
	}
	size_t MatrixF::totalSize() {
		return sizeof(float)*_size;
	}
	size_t MatrixF::size() {
		return _size;
	}

	MatrixU16::Matrix<uint16_t>(size_t size) : Array(TYPE_UINT16){
		_size = size;
		allocate(_size);
	}
	void* MatrixU16::data() {
		return static_cast<uint16_t*>(getData());
	}
	void MatrixU16::setData(uint16_t d, int index) {
		_data[index] = d;
	}
	MatrixU16::~Matrix() {
		delete[] _data;
	}
	uint16_t MatrixU16::get(int index) {
		return _data[index];
	}
	size_t MatrixU16::totalSize() {
		return sizeof(uint16_t)*_size;
	}
	size_t MatrixU16::size() {
		return _size;
	}

	MatrixU8::Matrix<uint8_t>(size_t size) : Array(TYPE_UINT8){
		_size = size;
		allocate(_size);
	}
	void* MatrixU8::data() {
		return static_cast<uint8_t*>(getData());
	}
	void MatrixU8::setData(uint8_t d, int index) {
		_data[index] = d;
	}
	MatrixU8::~Matrix() {
		delete[] _data;
	}
	uint8_t MatrixU8::get(int index) {
		return _data[index];
	}
	size_t MatrixU8::totalSize() {
		return sizeof(uint8_t)*_size;
	}
	size_t MatrixU8::size() {
		return _size;
	}

	MatrixL::Matrix<long>(size_t size) : Array(TYPE_UINT8){
		_size = size;
		allocate(_size);
	}
	void* MatrixL::data() {
		return static_cast<long*>(getData());
	}
	void MatrixL::setData(long d, int index) {
		_data[index] = d;
	}
	MatrixL::~Matrix() {
		delete[] _data;
	}
	long MatrixL::get(int index) {
		return _data[index];
	}
	size_t MatrixL::totalSize() {
		return sizeof(long)*_size;
	}
	size_t MatrixL::size() {
		return _size;
	}
}
