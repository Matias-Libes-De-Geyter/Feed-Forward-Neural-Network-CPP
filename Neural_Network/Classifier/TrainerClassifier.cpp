#include "TrainerClassifier.hpp"


// ======== TRAINER CLASSIFIER ======== //
TrainerClassifier::TrainerClassifier(FFNN& model, const hyperparameters& hyper) : _model(model), _hyper(hyper) {
	_scope = nullptr;
	_xtrain = nullptr;
	_ytrain = nullptr;
	_xvalid = nullptr;
	_yvalid = nullptr;
}

void TrainerClassifier::set_scope(Scope& scope) {
	_scope = &scope;
}

void TrainerClassifier::set_data(Dataset& train, Dataset& validation) {
	_xtrain = &train.x;
	_ytrain = &train.y;

	_xvalid = &validation.x;
	_yvalid = &validation.y;
}

void TrainerClassifier::run(bool store) {

	// Store data init
	d_vector CELoss;
	d_vector train_acc_array;
	d_vector val_acc_array;

	// Early stopping
	FFNN best(_hyper);
	int iPatience = 0;
	double bestLoss = 2;
	int nb_epochs = _hyper.max_epochs;

	const int n_batches = _hyper.n_train_samples / _hyper.mini_batch_size;
	for (int epoch = 0; epoch < nb_epochs; epoch++) {

		double epoch_loss = 0;
		int train_correct = 0;
		int val_correct = 0;

		// Train accuracy
		for (int n = 0; n < n_batches; n++) {
			Matrix& X = (*_xtrain)[n];
			Matrix& Y = (*_ytrain)[n];

			_model.forward(X, true);
			_model.backpropagation(X, Y);

			_scope->step(_model);

			// Loss & accuracy
			const Matrix& y_pred = _model.getOutput();
			epoch_loss += CELossFunction(y_pred, Y);

			Matrix y_pred_one_hot = y_pred.setMaxToOne();
			for (int i = 0; i < Y.rows(); i++)
				if (Y.row(i) == y_pred_one_hot.row(i))
					train_correct++;
		}
		epoch_loss /= n_batches;
		double train_accuracy = 100.0 * train_correct / _hyper.n_train_samples;

		// Validation accuracy
		for (int n = 0; n < _hyper.n_val_samples; n++) {
			Matrix& X = (*_xvalid)[n];
			Matrix& Y = (*_yvalid)[n];

			_model.forward(X, false);
			
			const Matrix& y_pred = _model.getOutput();
			Matrix y_pred_one_hot = y_pred.setMaxToOne();
			if (Y.row(0) == y_pred_one_hot.row(0))
				val_correct++;
		}
		double val_accuracy = 100.0 * val_correct / _hyper.n_val_samples;

		// Printing the results
		print("[Epoch ", epoch+1, "/", _hyper.max_epochs, "] ",
			  "Loss = ", epoch_loss, " | ",
			  "train_acc = ", train_accuracy, " % | ",
			  "val_acc = ", val_accuracy, " %");

		// Storing data
		if (store) {
			// Accuracy
			train_acc_array.push_back(train_accuracy);
			val_acc_array.push_back(val_accuracy);

			// Loss
			CELoss.push_back(epoch_loss);
		}

		// Implement early stopping
		if (_hyper.early_stopping) {
			if (epoch_loss < bestLoss) {
				bestLoss = epoch_loss;
				iPatience = 0;
			} else iPatience++;

			if (iPatience > _hyper.patience) {
				best.copyLayers(_model);
				nb_epochs = epoch;
				print("Breaking"); break;
			}
		}
	}
	if(store)
		writeFile(train_acc_array, val_acc_array, CELoss, nb_epochs, "training_data.csv");
}