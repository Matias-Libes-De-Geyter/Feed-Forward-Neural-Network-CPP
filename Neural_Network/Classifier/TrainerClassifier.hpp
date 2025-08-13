#include "..\Classifier/Scope.hpp"
#include "..\Dataset/Dataset.hpp"


#ifndef TRAINER_HPP
#define TRAINER_HPP


// ======== TRAINER CLASSIFIER ======== //
class TrainerClassifier {
private:
	const hyperparameters _hyper;
	FFNN& _model;

	Scope* _scope;
	std::vector<Matrix>* _xtrain;
	std::vector<Matrix>* _ytrain;
	std::vector<Matrix>* _xvalid;
	std::vector<Matrix>* _yvalid;

public:
	TrainerClassifier(FFNN&, const hyperparameters&);
	void set_scope(Scope&);
	void set_data(Dataset&, Dataset&);
	void run(bool);
};

#endif