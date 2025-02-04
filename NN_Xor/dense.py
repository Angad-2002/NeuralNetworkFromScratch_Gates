from layer import Layer
import numpy as np

class Dense(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn (output_size, input_size)
        self.bias = np.random.randn (output_size, 1)

    def forward(self, input):
        self.input = input
        return self.bias + np.dot (self.weights, self.input)

    def backward(self, output_g, l_rate):
        weights_g = np.dot (output_g, self.input.T)
        self.weights -= (l_rate * weights_g)
        self.bias -= (l_rate * output_g)
        return np.dot (self.weights.T, output_g)
    
