#include "DenseBlock.hpp"


// ======== DENSE LAYER ======== //
DenseBlock::DenseBlock(const int& n_inputs, const int& n_neurons) : m_weights(n_inputs + 1, n_neurons) {

	double limit = std::sqrt(6.0 / (n_inputs + n_neurons));

	for (size_t i = 0; i < n_neurons; i++)
		for (size_t j = 0; j < n_inputs + 1; j++)
			m_weights(j, i) = random(-limit, limit);
}

void DenseBlock::forward(const Matrix& inputs, ActivationType activation) {

	// Y = X * W
	MATRIX_OPERATION::compute_Y_from_input(m_Y, inputs, m_weights);

	// Z = a(Y)
	switch (activation) {
	case ActivationType::ReLU:
		m_Z = ACTIVATION::ReLU_activation(m_Y);
		break;
	case ActivationType::Softmax:
		m_Z = ACTIVATION::softmax_activation(m_Y);
		break;
	}
};