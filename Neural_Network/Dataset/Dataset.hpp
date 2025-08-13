#include "..\Utilities/functions.hpp"


#pragma once

#ifndef DATASET_HPP
#define DATASET_HPP



// ======== DATASET LOADER ======== //
inline int reverseInt(int i) { // To little-endian
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
inline void readMNIST(const std::string& imageFile, const std::string& labelFile,
	d_matrix& images, d_vector& labels) {
	std::ifstream imgFile(imageFile, std::ios::binary);
	std::ifstream lblFile(labelFile, std::ios::binary);

	int magicNumber, numImages, numRows, numCols;
	imgFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
	magicNumber = reverseInt(magicNumber);
	imgFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
	numImages = reverseInt(numImages);
	imgFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
	numRows = reverseInt(numRows);
	imgFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
	numCols = reverseInt(numCols);

	int labelMagicNumber, numLabels;
	lblFile.read(reinterpret_cast<char*>(&labelMagicNumber), sizeof(labelMagicNumber));
	labelMagicNumber = reverseInt(labelMagicNumber);
	lblFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
	numLabels = reverseInt(numLabels);

	int n = std::min(numImages, numLabels);
	images.resize(n, std::vector<double>(numRows * numCols));
	labels.resize(n);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < numRows * numCols; ++j) {
			unsigned char pixel;
			imgFile.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
			images[i][j] = static_cast<double>(pixel) / 255.0;
		}
		unsigned char label;
		lblFile.read(reinterpret_cast<char*>(&label), sizeof(label));
		labels[i] = static_cast<double>(label);
	}
}


// ======== DATASET ======== //
struct Dataset {
	std::vector<Matrix> x;
	std::vector<Matrix> y;
};
inline Dataset DataLoader(const hyperparameters& hyper, const std::string& dataset_type) {

	int n_iter = 0;
	int batch_size = 0;
	d_matrix images;
	d_vector labels;
	std::string ImagesFile;
	std::string LabelsFile;
	if (dataset_type == "train") {
		ImagesFile = "executable/database/MNIST/train-images.idx3-ubyte";
		LabelsFile = "executable/database/MNIST/train-labels.idx1-ubyte";
		n_iter = hyper.n_train_samples / hyper.mini_batch_size;
		batch_size = hyper.mini_batch_size;
	}
	else if (dataset_type == "validation") {
		ImagesFile = "executable/database/MNIST/t10k-images.idx3-ubyte";
		LabelsFile = "executable/database/MNIST/t10k-labels.idx1-ubyte";
		n_iter = hyper.n_val_samples;
		batch_size = 1;
	}
	else
		print("Dataset type is wrong");

	readMNIST(ImagesFile, LabelsFile, images, labels);
	d_matrix labels_hotOnes = hotOne(labels, 10);

	Dataset data;
	data.x.reserve(n_iter);
	data.y.reserve(n_iter);

	for (int n = 0; n < n_iter; n++) {
		d_matrix x_train(&images[batch_size * n], &images[batch_size * (n + 1)]);
		d_matrix y_train(&labels_hotOnes[batch_size * n], &labels_hotOnes[batch_size * (n + 1)]);

		Matrix X(x_train);
		Matrix Y(y_train);

		data.x.emplace_back(std::move(X));
		data.y.emplace_back(std::move(Y));
	}

	return data;
};

#endif