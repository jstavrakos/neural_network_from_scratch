from layer import Layer
from utility import sigmoid, sigmoid_derivative, transpose


class Activation_Layer(Layer):
    def __init__(self):
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_derivative(self.input) * output_error
