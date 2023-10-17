# abstract base class: inheriting classes must implement forward and backward prop
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass