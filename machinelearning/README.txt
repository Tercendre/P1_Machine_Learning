If the following runs and you see the below window pop up where a line segment spins in a circle, you can skip to the pytorch instillation steps. You should use the conda environment for this since conda comes with the libraries we need.


python autograder.py --check-dependencies


you will need to install the following libraries:

numpy matplotlib pytorch


If your setup is different, you can refer to numpy and matplotlib installation instructions. 

After installing, try the dependency check.

python autograder.py --check-dependencies


Here's a summary of the Project you can check ist advancement with the autograder commmand bellow : 

Question 1: Linear Regression
Implement a linear regression model that fits a straight line to given data points by completing the model’s initialization, prediction, loss calculation using mean squared error, and training with gradient descent updates.

python autograder.py -q q1

Question 2: Non-linear Regression
Train a neural network to approximate the function sin(x) over the interval [-2π, 2π] by implementing a simple regression model using mean squared error loss and gradient-based training.

python autograder.py -q q2

Question 3: Digit Classification
Build a neural network to classify handwritten digits from MNIST, outputting scores for each digit class and training the model using cross-entropy loss to achieve at least 97% test accuracy.

python autograder.py -q q3

Question 4: Language Identification
Implement a recurrent neural network (RNN) that processes variable-length input words character-by-character to classify the language of each word, aiming for at least 81% test accuracy.

python autograder.py -q q4

Question 5: Convolutional Neural Networks
Implement a 2D convolution function and a convolutional neural network model that applies convolutions to image data before classification, targeting 80% accuracy on a simplified MNIST subset.

python autograder.py -q q5

Question 6: Attention
Implement scaled dot-product attention with a causal mask that restricts the model from attending to future inputs, forming a core component for sequence modeling tasks.

python autograder.py -q q6

