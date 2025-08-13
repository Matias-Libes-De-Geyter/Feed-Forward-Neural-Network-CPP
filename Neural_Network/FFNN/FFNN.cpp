#include "FFNN.hpp"


// ======== NEURAL NETWORK ======== //
FFNN::FFNN(const hyperparameters& hyper) : _hyper(hyper) {

	// Getting the input and output dimensions into layer_sizes
	d_vector layer_sizes = _hyper.hidden_layer_sizes;
	layer_sizes.insert(layer_sizes.begin(), _hyper.input_dim);
	layer_sizes.push_back(_hyper.output_dim);
	L = layer_sizes.size() - 1;

	// Getting the vectors ready
	m_dW.resize(L);
	m_dZ.resize(L);
	m_layers.clear();
	m_layers.reserve(L);
	for (int l = 0; l < L; l++)
		m_layers.emplace_back(layer_sizes[l], layer_sizes[l + 1]);
}

void FFNN::forward(Matrix& input, const bool learning) {

	// Add dropout only when the FFNN is learning. Activate with softmax only if it's the last layer.
	m_layers[0].forward(input);
	for (int l = 1; l < L; l++) {
		ActivationType activation = (l == L - 1) ? ActivationType::Softmax : ActivationType::ReLU;
		Matrix input_next = learning ? m_layers[l - 1].output().dropoutMask(_hyper.dropout_rate) : m_layers[l - 1].output();
		m_layers[l].forward(input_next, activation);
	}

	m_output = m_layers.back().output();
}

void FFNN::backpropagation(Matrix& input, const Matrix& y_real) {

	// Last layer of backprop
	m_dZ[L - 1] = m_layers[L - 1].output() - y_real;
	MATRIX_OPERATION::compute_dW_from_input(m_dW[L - 1], m_layers[L - 2].output(), m_dZ[L - 1]);

	// Recurrent backprop
	for (int l = L - 2; l >= 0; l--) {
		MATRIX_OPERATION::compute_dZ_from_next(m_dZ[l], m_dZ[l + 1], m_layers[l + 1].weights(), m_layers[l].preactivation());
		MATRIX_OPERATION::compute_dW_from_input(m_dW[l], (l == 0 ? input : m_layers[l - 1].output()), m_dZ[l]);
	}
}

void FFNN::saveWeights(const std::string& filename) {
    std::ofstream file(filename);
    for (auto& layer : m_layers) {
		Matrix W = layer.weights();
		for (size_t i = 0; i < W.rows(); i++) {
			for (size_t j = 0; j < W.cols(); j++)
				file << W(i, j) << " ";
			file << "\n";
		}
		file << "===\n";
    } file.close();
}
void FFNN::loadWeights(const std::string& filename) {
    std::ifstream file(filename); std::string line;
    int layer_index = 0; d_matrix W;
    while (std::getline(file, line)) {
        if (line == "===") {
            m_layers[layer_index].setWeights(W);
            W.clear();
            layer_index++;
        }
        else {
            std::istringstream iss(line);
            d_vector row;
            double val;
            while (iss >> val)
                row.push_back(val);
            W.push_back(row);
        }
    } file.close();
}