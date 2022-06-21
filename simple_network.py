import random
import numpy as np
from typing import List


class Network:
    def __init__(self, network_definition: List[int]):
        """
            Biases and weights are generated randomly using Gaussian distribution.
        """
        self.number_of_layers = len(network_definition)
        self.network_definition = network_definition
        self.biases = [np.random.randn(y,1) for y in network_definition[1:]]
        self.weights = [np.random.randn(y, x) for x, y in 
            zip(network_definition[:-1], network_definition[1:])]


    def _feed_forward(self, a: int):
        """
            Apply function a' = sig(wa + b) to each layer
        """
        for b, w in zip(self.biases, self.weights):
            a = self._sigmoid(np.dot(w,a)+b)
        return a

    def _sigmoid(self, z: np.ndarray) -> float:
        return 1.0/(1.0+np.exp(-z))

    def _sigmoid_prime(self, z: np.ndarray ):
        return self._sigmoid(z)*(1-self._sigmoid(z))

    def _backprop(self, x, y):
            """Return a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``self.biases`` and ``self.weights``."""
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = self._sigmoid(z)
                activations.append(activation)
            
            # backward pass
            delta = self.cost_derivative(activations[-1], y) * \
                self._sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())

            for l in range(2, self.number_of_layers):
                z = zs[-l]
                sp = self._sigmoid_prime(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self._feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def _update_mini_batch(self, mini_batch, learning_rate: float):
        """Update weights and biases by applying gradient descent using backpropagation 
        to a single mini batch. The "mini_batch" is a list of tuples (x, y)."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb 
                    for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_set, number_of_epochs: int, batch_size: int, learning_rate: float, test_data=None):
        """
            test_data - if supplied is used evaluate progress after each epoch
        """
        if test_data: 
            n_test = len(test_data)

        for j in range(number_of_epochs):
            random.shuffle(training_set)
            mini_batches = [training_set[k:k+batch_size]for k in range(0, len(training_set), batch_size)]

            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, learning_rate)
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))