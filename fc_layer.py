from layer import Layer
import numpy as np
from utility import initialize_weights, initialize_biases, transpose


class FC_Layer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = initialize_weights(input_size, output_size)
        self.biases = initialize_biases(output_size)

    def forward_propagation(self, input_data):
        self.input = input_data  # from the base class
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    # computes derivative of error with respect to inputs (dE/dx) and to weights (dE/dw). returns the input_error for back prop
    # we are able to pass in the output error, because that comes from the layer "after" it
    def backward_propagation(self, output_error, learning_rate):
        # dE/dx = dE/dy * (weights transpose)
        input_error = np.dot(output_error, self.weights.T)
        # dE/dw = (x transpose) * dE/dy
        weights_error = np.dot(self.input.T, output_error)

        # update the parameters accordingly
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error
