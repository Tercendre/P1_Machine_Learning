
# P1\_Machine\_Learning

This project involves implementing and training several machine learning and deep learning models in Python, primarily using NumPy, Matplotlib, and PyTorch, to solve various tasks such as regression, image classification, and language identification. I didn't implement the autograder, it is used by our administration to test our projects.

Feel free to download the zip file and follow the different commands described below to understand the purpose of the different functions implemented.

If you were to look at the code, the most important implementations are in:

"Models.py" – Perceptron and neural network models for a variety of applications.

## README PROJECT P1\_Machine\_Learning

You will need to install the following libraries in your environment. We recommend using a conda environment as it contains most of the libraries we’ll need.

Libraries: numpy, matplotlib, pytorch

Explained below:

If the following runs and you see the window pop up where a line segment spins in a circle, you can skip to the PyTorch installation steps. You should use the conda environment for this since conda comes with the libraries we need.

Run the command: python autograder.py --check-dependencies

If your setup is different, you can refer to the NumPy and Matplotlib installation instructions.

After installing those, try the dependency check again using: python autograder.py --check-dependencies

Here’s a summary of the project and some commands to follow its progress:

Question 1: Linear Regression
Implement a linear regression model that fits a straight line to given data points by completing the model’s initialization, prediction, loss calculation using mean squared error, and training with gradient descent updates.
Command: python autograder.py -q q1

Question 2: Non-linear Regression
Train a neural network to approximate the function sin(x) over the interval \[-2π, 2π] by implementing a simple regression model using mean squared error loss and gradient-based training.
Command: python autograder.py -q q2

Question 3: Digit Classification
Build a neural network to classify handwritten digits from MNIST, outputting scores for each digit class and training the model using cross-entropy loss to achieve at least 97% test accuracy.
Command: python autograder.py -q q3

Question 4: Language Identification
Implement a recurrent neural network (RNN) that processes variable-length input words character-by-character to classify the language of each word, aiming for at least 81% test accuracy.
Command: python autograder.py -q q4

Question 5: Convolutional Neural Networks
Implement a 2D convolution function and a convolutional neural network model that applies convolutions to image data before classification, targeting 80% accuracy on a simplified MNIST subset.
Command: python autograder.py -q q5

Question 6: Attention
Implement scaled dot-product attention with a causal mask that restricts the model from attending to future inputs, forming a core component for sequence modeling tasks.
Command: python autograder.py -q q6
