# pylint: disable=bad-whitespace
from network import Network
import numpy as np
from fc_layer import FC_Layer
from activation_layer import Activation_Layer

LEARNING_RATE = 0.1
EPOCHS = 10000

def main():
    '''
    # Inverter
    x_train = np.array([[[0]], [[1]]])
    y_train = np.array([[[1]], [[0]]])

    test_network = Network()
    test_network.add(FC_Layer(1,1))
    test_network.add(Activation_Layer())

    test_network.fit(x_train, y_train, EPOCHS, LEARNING_RATE)
    output = test_network.predict(x_train)
    print(output)
    '''
    # XOR
    print(f"XOR Training with {EPOCHS} epochs and a learning rate of {LEARNING_RATE}")
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # create a network with multiple layers
    nn = Network()
    nn.add(FC_Layer(2, 3)) # 2 initial inputs, 3 outputs in hidden layer
    nn.add(Activation_Layer())
    nn.add(FC_Layer(3, 1)) # compress 3 hidden layer inputs into 1 output
    nn.add(Activation_Layer())

    # train the neural net
    nn.fit(x_train, y_train, EPOCHS, LEARNING_RATE)

    # test predictions
    output = nn.predict(x_train)
    for val in output:
        print(val)

    #-------------------------------------------------------------------------------------------------------------------------------------
    # Two bit adder
    print(f"Two Bit Adder Training with {EPOCHS} epochs and a learning rate of {LEARNING_RATE}")
    #x_train2 = np.array([[[0, 0, 0]], [[0, 0, 1]], [[0, 1, 0]], [[1, 0, 0]], [[0, 1, 1]], [[1, 0, 1]], [[1, 1, 0]], [[1, 1, 1]]])
    #y_train2 = np.array([[[0, 0]], [[0, 1]], [[0, 1]], [[0, 1]], [[1, 0]], [[1, 0]], [[1, 0]], [[1, 1]]])
    # a1, a0, b1, b0, carry in
    x_train2 = np.array([[[0,0,0,0,0]], [[0,0,0,0,1]], [[0,0,0,1,0]], #[[0,0,1,0,0]],
                         [[0,1,0,0,0]], [[1,0,0,0,0]],
                         
                         [[0,0,0,1,1]], [[0,0,1,0,1]], [[0,1,0,0,1]], #[[1,0,0,0,1]], 
                         [[0,0,1,1,0]], [[0,1,0,1,0]], [[1,0,0,1,0]], [[0,1,1,0,0]], 
                         [[1,0,1,0,0]], [[1,1,0,0,0]],

                         [[0,0,1,1,1]], [[0,1,0,1,1]], [[0,1,1,0,1]], #[[0,1,1,1,0]],
                         [[1,0,0,1,1]], [[1,0,1,0,1]], [[1,0,1,1,0]], [[1,1,0,0,1]],
                         [[1,1,0,1,0]], [[1,1,1,0,0]], 
                         
                         [[0,1,1,1,1]], [[1,0,1,1,1]], [[1,1,0,1,1]], #[[1,1,1,0,1]],
                         [[1,1,1,1,0]], [[1,1,1,1,1]],
                        ])
    # carry out, s1, s0
    y_train2 = np.array([[[0,0,0]], [[0,0,1]], [[0,0,1]], #[[0,1,0]],
                         [[0,0,1]], [[0,1,0]],

                         [[0,1,0]], [[0,1,1]], [[0,1,0]], #[[0,1,1]],
                         [[0,1,1]], [[0,1,0]], [[0,1,1]], [[0,1,1]],
                         [[1,0,0]], [[0,1,1]],

                         [[1,0,0]], [[0,1,1]], [[1,0,0]], #[[1,0,0]],
                         [[1,0,0]], [[1,0,1]], [[1,0,1]], [[1,0,0]],
                         [[1,0,0]], [[1,0,1]],
                         
                         [[1,0,1]], [[1,1,0]], [[1,0,1]], #[[1,1,0]],
                         [[1,1,0]], [[1,1,1]],
                        ])

    # create a network with multiple layers
    nn2 = Network()
    nn2.add(FC_Layer(5, 10)) # 5 initial inputs, 10 outputs in hidden layer
    nn2.add(Activation_Layer())
    nn2.add(FC_Layer(10, 3)) # compress 10 hidden layer inputs into 3 outputs
    nn2.add(Activation_Layer())
    #nn2.add(FC_Layer(5, 3))
    #nn2.add(Activation_Layer())

    # train the neural net
    nn2.fit(x_train2, y_train2, EPOCHS, LEARNING_RATE)

    # test predictions
    x_test = np.array([[[0,0,1,0,0]], [[1,0,0,0,1]], [[0,1,1,1,0]], [[1,1,1,0,1]]])
    #x_test_answers = np.array([[[0,1,0]], [[0,1,1]], [[1,0,0]], [[1,1,0]]])
    output2 = nn2.predict(x_test)
    for arr in output2:
        print(arr)


if __name__ == '__main__':
    main()

