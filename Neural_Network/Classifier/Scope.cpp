#include "Scope.hpp"


// ======== SCOPE ======== //
Scope::Scope(FFNN& model, const hyperparameters& hyper) : _hyper(hyper), t(1) {

	auto params = model.getParameters();
	for (auto& [W, dW] : params) {
		Matrix emptyWeight(W->rows(), W->cols());
		M.emplace_back(emptyWeight);
		V.emplace_back(emptyWeight);
	}
}

void Scope::Adam(Matrix& W, Matrix& dW, const int k) {

	const double beta_1 = 0.9;
	const double beta_2 = 0.999;

	M[k] = M[k] * beta_1 + dW * (1 - beta_1);
	V[k] = V[k] * beta_2 + dW.hadamard(dW) * (1 - beta_2);

	double bias_1_correction = 1.0 - std::pow(beta_1, t);
	double bias_2_correction = 1.0 - std::pow(beta_2, t);

	for (size_t i = 0; i < W.rows(); i++) {
		for (size_t j = 0; j < W.cols(); j++) {
			double M_hat = M[k](i, j) / bias_1_correction;
			double V_hat = V[k](i, j) / bias_2_correction;
			W(i, j) -= (M_hat / (sqrt(V_hat) + 1e-8)) * _hyper.learning_rate;
		}
	}
}

void Scope::SGD(Matrix& W, Matrix& dW) {

	W -= dW * _hyper.learning_rate;
}