#include <initializer_list>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <cmath>


#ifndef MATRIX_H
#define MATRIX_H

using d_vector = std::vector<double>;
using d_matrix = std::vector<std::vector<double>>;

class Matrix {
private:
	size_t _rows;
	size_t _cols;

	d_vector _matrix;

public:
	size_t rows() const { return _rows; };
	size_t cols() const { return _cols; };

	// Constructors
	inline Matrix() {};
	Matrix(std::vector<std::vector<double>>);
	Matrix(std::initializer_list<std::initializer_list<double>>);
	inline Matrix(const double a) : _rows(1), _cols(1), _matrix(d_vector(1, a)) {};
	inline Matrix(size_t row, size_t columns) : _rows(row), _cols(columns), _matrix(d_vector(_rows * _cols, 0.0)) {};

	// Operators
	Matrix& operator=(std::initializer_list<std::initializer_list<double>>);
	Matrix operator*(const Matrix& B) const;
	Matrix operator*(const double b) const;
	Matrix& operator*=(const double b);
	Matrix hadamard(const Matrix& B) const;
	const bool operator==(const Matrix& B) const;

	Matrix operator+(const Matrix& B) const;
	Matrix& operator+=(const Matrix& B);
	Matrix operator-(const Matrix& B) const;
	Matrix& operator-=(const Matrix& B);

	// Modification
	Matrix T() const;
	Matrix addBias() const;
	Matrix addBias_then_T() const;
	Matrix removeBias() const;
	Matrix T_then_removeBias() const;
	Matrix dropoutMask(double dropout) const;
	Matrix setMaxToOne() const;

	void fill(const double b);
	int getMaxIndex() const;

	// Getting data
	inline double& operator()(size_t idx) { return _matrix[idx]; };
	inline const double& operator()(size_t idx) const { return _matrix[idx]; };
	inline double& operator()(size_t i, size_t j) { return _matrix[i * _cols + j]; };
	inline const double& operator()(size_t i, size_t j) const { return _matrix[i * _cols + j]; };
	inline Matrix getParams() const { return Matrix{ {static_cast<double>(_rows), static_cast<double>(_cols)} }; };
	inline std::vector<double> row(int i) { return std::vector<double>(_matrix.begin() + i * _cols, _matrix.begin() + (i + 1) * _cols); };
};

#endif