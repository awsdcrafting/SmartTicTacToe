import random

import numpy as np
from scipy.special import expit

mutation_rate = 0.03
mutation_range = 0.5


class NeuralNetwork:

    def __init__(self, structure, initialize_random=True):

        self.structure = []
        self.weights = []
        self.bias = []

        self.score = 0

        self.structure = structure

        if initialize_random:
            # Creating weights
            for layer_id in range(0, len(structure) - 1):
                self.create_weights(layer_id)

            # Creating bias
            for layer_id in range(1, len(structure)):
                self.create_bias(layer_id)

    def calculate(self, network_input):
        last_result = network_input
        for layer_id in range(0, len(self.bias)):
            last_result = expit(np.add(self.weights[layer_id].dot(last_result), self.bias[layer_id]))
        return last_result

    def create_weights(self, layer_id):
        self.weights.append(np.random.uniform(-1, 1, [self.structure[layer_id + 1], self.structure[layer_id]]))

    def create_bias(self, layer_id):
        np.random.seed(random.randint(0, 80))
        new_bias = np.random.uniform(-1, 1, [self.structure[layer_id], 1])
        self.bias.append(new_bias)

    def print_weights(self):
        print("Weights:")
        for layer_id in range(0, len(self.weights)):
            print("Layer-ID: {}".format(layer_id))
            print(self.weights[layer_id])
        print()

    def print_bias(self):
        print("Bias:")
        for layer_id in range(0, len(self.bias)):
            print("Layer-ID: {}".format(layer_id))
            print(self.bias[layer_id])
        print()

    def print_network(self):
        print("Network:")
        self.print_weights()
        self.print_bias()
        print("\n")


def crossover(network1, network2):
    new_network = NeuralNetwork(structure=network2.structure, initialize_random=False)

    network1_weights = []
    network2_weights = []

    for a in network1.weights:
        for ix, iy in np.ndindex(a.shape):
            network1_weights.append(a[ix, iy])
    for b in network2.weights:
        for ix, iy in np.ndindex(b.shape):
            network2_weights.append(b[ix, iy])

    network1_bias = []
    network2_bias = []

    for a in network1.bias:
        for ix, iy in np.ndindex(a.shape):
            network1_bias.append(a[ix, iy])
    for b in network2.bias:
        for ix, iy in np.ndindex(b.shape):
            network2_bias.append(b[ix, iy])

    weight_divider = random.randint(0, len(network1_weights) - 1)
    new_weights = network1_weights.copy()[:weight_divider]
    new_weights.extend(network2_weights[weight_divider:])

    bias_divider = random.randint(0, len(network1_bias) - 1)
    new_bias = network1_bias.copy()[:bias_divider]
    new_bias.extend(network2_bias[bias_divider:])

    # Mutating Weights
    for current_weight in range(0, len(new_weights)):
        if random.uniform(0, 1) < mutation_rate:
            new_weights[current_weight] = new_weights[current_weight] + random.uniform(-mutation_range,
                                                                                       + mutation_range)

    # Mutating Bias
    for current_bias in range(0, len(new_bias)):
        if random.uniform(0, 1) < mutation_rate:
            new_bias[current_bias] = new_bias[current_bias] + random.uniform(-mutation_range, + mutation_range)

    start = 0
    for layer_id in range(0, len(network1.structure) - 1):
        stop = start + network1.structure[layer_id] * network1.structure[layer_id + 1]
        new_network.weights.append(np.reshape(
            new_weights[start:stop],
            [network1.structure[layer_id + 1], network1.structure[layer_id]]))
        start = stop

    start = 0
    for layer_id in range(1, len(network1.structure)):
        stop = start + network1.structure[layer_id]
        new_network.bias.append(np.reshape(
            new_bias[start:stop],
            [network1.structure[layer_id], 1]))

    return new_network
