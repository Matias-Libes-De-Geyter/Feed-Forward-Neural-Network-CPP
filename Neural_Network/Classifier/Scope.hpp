#include "..\FFNN/FFNN.hpp"

#ifndef SCOPE_HPP
#define SCOPE_HPP


// ======== SCOPE ======== //
class Scope {
private:
	const hyperparameters& _hyper;

	std::vector<Matrix> M, V;

	int t;

public:
	Scope(FFNN&, const hyperparameters&);

	void Adam(Matrix& W, Matrix& dW, const int k);
	void SGD(Matrix& W, Matrix& dW);

	inline void step(FFNN& model) {

		int k = 0;
		for (auto& [W, dW] : model.getParameters())
			Adam(*W, *dW, k++);

		t++;

	};

};

#endif