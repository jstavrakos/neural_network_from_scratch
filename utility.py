import numpy as np

# choose from uniform distribution
'''
def initialize_weights(rows, cols):
    return np.random.rand(rows, cols) - 0.5 # a random initial weight from [-0.5, 0.5)
def initialize_biases(cols):
    return np.random.rand(1, cols) - 0.5 # a random inital bias from [-0.5, 0.5)
'''

# choose from normal distrubution
def initialize_weights(rows, cols):
    return np.random.normal(0, 0.5, size=(rows, cols))

def initialize_biases(cols):
    return np.random.normal(0, 0.5, size=(1, cols))

def transpose(matrix):
    transposed = [list(row) for row in zip(*matrix)]
    return transposed

# will be used as the activation function: o(x) 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# o'(x) = o(x) / (1 - o(x)) where o(x) is the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# the euclidean distance
def loss(u , v):
    if len(u) != len(v):
        raise ValueError("vectors must have the same length")

    squared_diff = [(u[i] - v[i]) ** 2 for i in range(len(u))]
    distance = sum(squared_diff) ** 0.5
    
    return distance

