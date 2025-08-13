#include "..\Utilities/functions.hpp"

#ifndef DB_HPP
#define DB_HPP

enum class ActivationType { ReLU, Softmax };


// ======== DENSEBLOCK ======== //
class DenseBlock {
private:
	Matrix m_weights;
	Matrix m_Y;
	Matrix m_Z;
	
	Matrix ReLU_activation(const Matrix& inputs);
	Matrix softmax_activation(const Matrix& inputs);

public:
	DenseBlock() : m_weights(), m_Y(), m_Z() {};
	DenseBlock(const int& n_inputs, const int& n_neurons);
	void forward(const Matrix& inputs, ActivationType activation = ActivationType::ReLU);

	inline void setWeights(const Matrix& weights) { m_weights = weights; };

	inline Matrix& weights() { return m_weights; };
	inline const Matrix& weights() const { return m_weights; };
	inline Matrix& preactivation() { return m_Y; };
	inline Matrix& output() { return m_Z; };
};

#endif