#include "functions.hpp"

// Random function
std::mt19937_64& get_rng() {
	static std::mt19937_64 rng{ std::random_device{}() };
	return rng;
}
double random(const double& min, const double& max) {
	return std::uniform_real_distribution<>{min, max}(get_rng());
}
int random_bit() {
	static std::uniform_int_distribution<int> dist(0, 1);
	return dist(get_rng());
}

d_matrix hotOne(const d_vector& y, const int& nElements) {

	d_matrix C(y.size(), d_vector(nElements, 0));

	for (int i = 0; i < y.size(); i++)
		C[i][y[i]] = 1;

	return C;
}

// Cross-entropy loss computation
double CELossFunction(const Matrix& y_pred, const Matrix& y_true) {
	double mean_loss = 0.0;

	for (int i = 0; i < y_pred.rows(); i++)
		for (int j = 0; j < y_pred.cols(); j++)
			if (y_true(i, j) == 1.0) {
				mean_loss += -log(std::max(y_pred(i, j), 1e-9));
				break;
			}

	mean_loss /= y_pred.rows();

	return mean_loss;
}


// Write output data to plot with python
void writeFile(const d_vector& train_acc, const d_vector& val_acc, const d_vector& loss, int nb_epochs, const std::string& filename) {
	std::ofstream outFile(filename);
	if (!outFile) {
		std::cerr << "Error opening file for writing: " << filename << std::endl;
		return;
	}

	outFile << "Epoch,TrainAcc,ValAcc,Loss\n";
	for (int epoch = 0; epoch < nb_epochs; ++epoch) {
		outFile << epoch + 1 << "," << train_acc[epoch] << "," << val_acc[epoch] << "," << loss[epoch] << "\n";
	}

	outFile.close();
}