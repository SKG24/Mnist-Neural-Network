# Mnist-Neural-Network
the implementation and training of a simple two-layer neural network on the MNIST digit recognizer dataset to illustrate the underlying mathematics of neural networks.

# Overview
The neural network architecture implemented is a simple feedforward network with:

An input layer with 784 nodes (28x28 pixel images flattened into a single vector).
A hidden layer with 10 nodes.
An output layer with 10 nodes, each representing one of the possible digits (0-9).
The process of building and training the neural network involves three main parts:

- Forward Propagation
- Backward Propagation
- Parameter Update

# Data Preparation
Load the MNIST Dataset: Each image is represented as a 784-element vector with pixel values between 0 and 255.
Transpose the Data: For convenience in matrix operations, each column represents an example, and each row represents a feature.
Divide Data into Training and Validation Sets: Separate the dataset into training and validation sets to avoid overfitting.

# Neural Network Initialization
Weights and Biases Initialization: Initialize weights and biases for each layer with small random values. This ensures the neural network starts with a diverse set of weights, promoting better learning.

Forward Propagation

- Input Layer: Directly take the input data (X).
- First Hidden Layer (Z1, A1):
Compute the linear combination of the inputs and weights plus bias (Z1 = W1 * X + B1).
Apply the ReLU activation function to introduce non-linearity (A1 = ReLU(Z1)).
- Output Layer (Z2, A2):
Compute the linear combination of the hidden layer activations and weights plus bias (Z2 = W2 * A1 + B2).
Apply the Softmax activation function to convert outputs into probabilities (A2 = Softmax(Z2)).

Backward Propagation

- Compute Error (Loss) at Output Layer:
Compare predictions (A2) with actual labels (Y) to compute the error (DZ2 = A2 - Y).
- Calculate Gradients for Weights and Biases:
Compute gradients for the output layer (DW2, DB2) using the error term and activations from the hidden layer.
Propagate the error back through the network to compute gradients for the first hidden layer (DZ1) and its weights and biases (DW1, DB1).
Apply the chain rule to account for the derivative of the activation function.

Parameter Update

-Gradient Descent:
Update weights and biases using the computed gradients and a predefined learning rate (alpha).
Repeat forward and backward propagation for a specified number of iterations, continuously updating the parameters to minimize the loss.

Activation Functions

- ReLU (Rectified Linear Unit):
Defined as ReLU(x) = max(0, x).
Introduces non-linearity to the model, allowing it to learn more complex functions.
- Softmax:
Converts raw scores from the output layer into probabilities, which sum to 1 across the output nodes.
Defined as Softmax(x) = e^x_i / Σ(e^x_j) for each output node i.
# One-Hot Encoding
Convert labels into a binary matrix representation, where the index of the true class is set to 1 and all other indices are 0. This is necessary for computing the error during backpropagation.
# Training the Network
- Initialize parameters.
- Perform forward propagation to compute predictions.
- Compute loss and perform backward propagation to calculate gradients.
- Update parameters using gradient descent.
- Iterate through the above steps for a defined number of iterations or until the model reaches a satisfactory accuracy.
# Validation and Testing
- After training, validate the neural network using the validation set to ensure it generalizes well to new, unseen data.
- Monitor accuracy and adjust hyperparameters such as the learning rate and the number of iterations as needed to improve performance.
