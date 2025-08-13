#include "Matrix.hpp"

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// Hyperparameters
struct hyperparameters {
	int input_dim;
	int output_dim;
	d_vector hidden_layer_sizes;
	double learning_rate;
	double dropout_rate;
	int max_epochs;
	int n_train_samples;
	int mini_batch_size;
	int n_val_samples;
	bool early_stopping;
	int patience;
};

std::mt19937_64& get_rng();
double random(const double& min, const double& max); // Random function
int random_bit(); // Random bit between 0 and 1

d_matrix hotOne(const d_vector& y, const int& nElements);
double CELossFunction(const Matrix& y_pred, const Matrix& y_true); // Return the cross-entropy loss.

// Utility function used in TrainerClassifier.h
void writeFile(const d_vector& train_acc, const d_vector& test_acc, const d_vector& loss, int nb_epochs, const std::string& filename);


namespace ACTIVATION {

	inline double deriv_ReLU(double inputs) {
		return (inputs > 0.0 ? 1.0 : 0);
	};

	inline Matrix ReLU_activation(const Matrix& inputs) {

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++)
				output(i, j) = std::max(0.0, inputs(i, j));
		return output;
	};

	inline Matrix softmax_activation(const Matrix& inputs) {

		d_vector maxs(inputs.rows());
		for (size_t i = 0; i < inputs.rows(); i++) {
			maxs[i] = inputs(i, 0);
			for (size_t j = 0; j < inputs.cols(); j++)
				if (inputs(i, j) > maxs[i])
					maxs[i] = inputs(i, j);
		}

		Matrix expvalues = inputs;
		d_vector sum_of_exps(inputs.rows(), 0);
		for (size_t i = 0; i < inputs.rows(); i++) {
			for (size_t j = 0; j < inputs.cols(); j++) {
				expvalues(i, j) = std::exp(inputs(i, j) - maxs[i]);
				sum_of_exps[i] += expvalues(i, j);
			}
		}

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++)
				output(i, j) = expvalues(i, j) / sum_of_exps[i];

		return output;
	};
}


namespace MATRIX_OPERATION {

	inline void compute_Y_from_input(Matrix& output, const Matrix& input, const Matrix& weights) {
		size_t output_rows = input.rows();
		size_t output_cols = weights.cols();
		size_t middle_dim = weights.rows();
		assert(middle_dim == input.cols() + 1);

		output = Matrix(output_rows, output_cols);
		for (size_t i = 0; i < output_rows; i++) {
			size_t row_offset = i * middle_dim;
			for (size_t k = 0; k < middle_dim - 1; k++) {
				double input_ik = input(i, k);
				for (size_t j = 0; j < output_cols; j++)
					output(i, j) += input_ik * weights(k, j);
			}
			size_t k = middle_dim - 1;
			double input_ik = input(i, k);
			for (size_t j = 0; j < output_cols; j++)
				output(i, j) += weights(k, j);
		}
	};

	inline void compute_dZ_from_next(Matrix& output, const Matrix& input, const Matrix& weights, const Matrix& preactivation) {
		const size_t batch = input.rows();
		const size_t next_cols = input.cols();
		const size_t weights_rows = weights.rows();
		const size_t cur_cols = weights_rows - 1;

		assert(next_cols == weights.cols());
		assert(preactivation.rows() == batch);
		assert(preactivation.cols() == cur_cols);

		output = Matrix(batch, cur_cols);
		for (size_t i = 0; i < batch; ++i) {
			const size_t row_offset_dZ = i * next_cols;
			const size_t row_offset_Y = i * cur_cols;
			for (size_t j = 0; j < cur_cols; ++j) {
				double sum = 0.0;
				for (size_t k = 0; k < next_cols; ++k)
					sum += input(row_offset_dZ + k) * weights(j, k);
				double deriv = ACTIVATION::deriv_ReLU(preactivation(row_offset_Y + j));
				output(i, j) = sum * deriv;
			}
		}
	};

	inline void compute_dW_from_input(Matrix& output, const Matrix& input, const Matrix& dZ) {
		const size_t batch = input.rows();
		const size_t output_rows = input.cols() + 1;
		const size_t output_cols = dZ.cols();

		assert(batch == dZ.rows());

		output = Matrix(output_rows, output_cols);
		for (size_t i = 0; i < batch; ++i) {
			const size_t row_offset_input = i * (output_rows - 1);
			const size_t row_offset_dZ = i * output_cols;

			for (size_t j = 0; j < output_rows - 1; ++j) {
				const double input_ij = input(row_offset_input + j);
				for (size_t k = 0; k < output_cols; ++k)
					output(j, k) += input_ij * dZ(row_offset_dZ + k);
			}
			for (size_t k = 0; k < output_cols; ++k)
				output(output_rows - 1, k) += dZ(row_offset_dZ + k);
		}
	};
}



// ===== PRINT FUNCTIONS =====
// Multiple print
template<typename... Args>
typename std::enable_if<(sizeof...(Args) > 1), void>::type
inline print(const Args&... args) { (std::cout << ... << args) << std::endl; }

// Single print
template<typename T>
typename std::enable_if<!std::is_same<T, Matrix>::value, void>::type
inline print(const T& arg) { std::cout << arg << std::endl; }

// Matrix print
inline void print(const Matrix& A) {
	const size_t rows = A.rows();
	const size_t cols = A.cols();

	std::ostringstream stream;
	stream << "[";
	for (size_t i = 0; i < rows; i++) {
		stream << "[";
		for (size_t j = 0; j < cols; j++) {
			stream << A(i, j);
			if (j < cols - 1) stream << ", ";
		}
		stream << "]";
		if (i < rows - 1) stream << ", " << std::endl;
	}
	stream << "]" << std::endl;

	std::cout << stream.str();
}

#endif