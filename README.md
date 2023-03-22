# Simple Neural Network with Keras

This is a simple Python script that creates a neural network using Keras and analyzes the loss and test accuracy. 

## Setup

The script requires the following libraries:
- Keras
- Numpy
- Matplotlib

## Functionality

The script performs the following steps:
1. Generates random input features and output labels.
2. Defines the model using Keras sequential API and adds a hidden layer of 10 neurons and another layer of 1 output neuron using ReLU activation function.
3. Compiles the model with specified loss function (mean squared error) and trains it using Adam optimizer.
4. Evaluates the model on test data using mean squared error and prints the test loss to the console.
5. Plots the training loss over time.
6. Plots predicted output against true output.

## Usage

To use the script, simply run it in a Python environment. It will output the sample input features and output labels and then generate and display two plots: the training loss over time and the predicted output against the true output.

## Credits

This script was created by [insert your name here].