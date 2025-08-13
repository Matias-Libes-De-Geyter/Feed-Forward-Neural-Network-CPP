# Feed-Forward-Neural-Network-CPP
### Model
- **Model:** Feed-Forward Neural Network,
- **Optimizer:** Adam optimizer,
- **Regularization methods:** Early stopping & Dropout.

## Introduction
The aim of this project was, using a very simple Feed-Forward Neural Network and Adam optimizer, to firstly get a good accuracy on MNIST database. This was a success, therefore the aim became to implement an interface so the user could draw numbers and ask the NN to guess. I used exclusively the slides of UToyko's Advanced Data Analysis class by *Takashi Ishida* (ishi@k.u-tokyo.ac.jp) for my code and understanding, with a hand from chatGPT for giving readMNIST, reverseInt (Big-endian to Little-endian conversion) and read/write files functions.

### Why C++ ?
- Firstly, I used C++ because I'm much more familiar with it than Python. I was also interested to challenge myself to see if I could build a neural network from scratch, without pytorch or similar libraries.
- Secondly, I used "SFML", a graphic library (with which I've done many projects), to ask the user to draw a number, and ask the FFNN to output its guess.
- Most importantly, C++ was chosen for it's compatibility with CUDA (which uses GPU), on which the code was translated. This is in the file "CUDA VERSION".

## Demo

Training & testing previews:

<p align="center">
  <img src="img/training.gif" width="185" />
  <img src="img/testing.gif" width="614" />
</p>


## Methodology
Firstly, I created the Matrix class to handle operations and vectors. From then on:
- Creation the DenseBlock class. I chose to put biases inside the weight matrices, and to initialise the weight matrices already transposed.
- Creation the FFNN class, and implementation of the forward pass with softmax activation at the end.
- Implementation of backpropagation using matrix operations and math.
- Optimization with Adam optimizer as detailed in the Scope class. I used the same constants, and added a small coefficient ```1e-8``` in the expression of $$w_{ij}^{l+1}$$, such that $$\frac{1}{\sqrt(\hat{v})} \longrightarrow \frac{1}{\sqrt(\hat{v}) + 10^{-8}}$$ in case $$\sqrt(\hat{v})$$ is null.
- Creation of TrainerClassifier to put everything together.

After implementing these classes and having a good accuracy on MNIST database, I've implemented the SFML library to create a drawing canvas.


### Hyperparameters
- I used a learning rate of $$10^{-3}$$. It gave good results compared to 0.01.
- There's two layers of 256 and 128 neurons.
- In Adam optimizer, $$\beta_m = 0.9$$ and $$\beta_v = 0.999$$.
- I used batches of **32 images**, and used it on the whole dataset. It worked better than 16 or 64. Since the **MNIST** dataset has 60000 training images, we only trained on $$1000$$ samples, during 30 epochs. A validation dataset was prepared with $$500$$ images.
- If early stopping is toggled, patience $$= 3$$.

## Results

### Observations
- Results on MNIST train database. When ran into the whole training database, the model gives the following results:
![Plots](img/output.png)

Here, the early stopper stopped the training after 20-or-so epochs. We can see the training accuracy, validation accuracy and training loss for each epochs.

- Results on MNIST test database:

After training on the whole train database, the model provides an **accuracy of $$98$$%** when tested on MNIST database's test files.

- In parallel, the code on CUDA ran two to four times faster than the basic C++ code. It is an interesting result that could be useful in the future.

### Discussion
- As we can see of the plots, the accuracy rises quite quickly, before settling.
- Also, the tests runs well on MNIST database, but when drawing numbers, the accuracy drops. This could be because the numbers of the database used for training are all centered, and that the way they were generated was different than mine. I tried to implement a gradient around the brush to fit the MNIST database-style and it gave better results, but it wasn't enough.

### Next steps
- I didn't implement flooding. It could improve the model.
- The next move would be to implement CNN to solve this problem. When writing by hand it doesn't give satisfying results (clearly above 50% accuracy but clearly below 70%).
- We could also translate matrices into vectors. This would be faster in C++ and also in Cuda since we currently need to flatten and expand matrices to use GPU acceleration.


---

## How to Use

- Run the ```FFNN.bat``` file. To train, press 'y'. Any other input would lead to the test interface.
- If training:
  - To plot the output of the training, run the ```plot.py``` file from the main folder.
- If testing:
  - Press "A" to get a guess, press "R" to reset the canvas.

To change the hyperparameters except boolean ```training```, you must recompile everything for now. The command to compile is: ```mingw32-make -f MakeFile```.



## Requirements

- Mingw32 compiler version ```gcc-14.2.0-mingw-w64ucrt-12.0.0-r2```.
- Python 3.x .

---

## Repository Structure

```plaintext
NeuralNetwork/
│
├── CUDA VERSION/       # The FFNN version exploiting CUDA features (x2-x4 computation time gain). No interface.
│
├── executable/
│   ├── database/       # Dataset
│   │   └── MNIST/
│   ├── main.exe            # Main executable
│   ├── model_weights.txt   # Save of the weights. Used to run the program without having to train it everytime
│   └── xxx.dll             # SFML and C++ Dlls used in the main.exe file.
│
├── img/
│   ├── testing.gif     # Training example
│   ├── training.gif    # Testing example
│   └── output.png
│
├── libs/          # SFML Library used for the window
│   ├── include/
│   └── lib/
│
├── Neural_Network/     # Main codes of the repository
│   ├── Blocks/
│   │   ├── DenseBlock.cpp
│   │   └── DenseBlock.hpp
│   ├── Classifier/
│   │   ├── TrainerClassifier.cpp
│   │   └── TrainerClassifier.hpp
│   │   ├── Scope.cpp
│   │   └── Scope.hpp
│   ├── Dataset/
│   │   └── Dataset.hpp
│   ├── FFNN/
│   │   ├── FFNN.cpp
│   │   └── FFNN.hpp
│   ├── Utilities/
│   │   ├── functions.cpp
│   │   ├── functions.hpp
│   │   ├── Matrix.cpp
│   │   └── Matrix.hpp
│   │
│   ├── main.cpp        # Main code that initiate all variables
│   └── plot.py         # Run "py Neural_Network/plot.py" to get a plot of the result of the training
│
├── MakeFile
├── FFNN.bat             # Execute this file to test the program
├── README.md           
└── training_data.csv   # Output from the training process, to plot the loss and accuracy
```

---