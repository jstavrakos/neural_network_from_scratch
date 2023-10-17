from utility import loss

class Network:
    def __init__(self):
        self.layers = []
        self.loss = loss

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        num_samples = len(x_train)

        # training loop
        for i in range(epochs):
            loss = 0
            for j in range(num_samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                loss += self.loss(y_train[j], output)

                # backward propagation
                # we must manually calculate the first dE/dy to start passing back through in back prop
                output_error = (output - y_train[j]) #* sigmoid_derivative(output)
                for layer in reversed(self.layers):
                    output_error = layer.backward_propagation(output_error, learning_rate)

            # calculate error --> mean of losses
            loss /= num_samples
            loss = sum(loss)
            if(i % 1000 == 0):
                print('epoch %d/%d   loss=%f' % (i, epochs, loss))
