import random as r

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def output(self, inputs):
        self.prediction = self.bias
        for index, i in enumerate(inputs, start=0):
            self.prediction += i * self.weights[index]
        
        return self.prediction

    def train(self, inputs: list, exp_out:list, learning_rate, epochs:int):
        needed_weights = len(inputs[0])
        for input in inputs:
            if len(input) != needed_weights:
                print("[!] Inputs with different lenghts were provided")
                raise Exception
        if needed_weights != len(self.weights):
            print("[!] Amount of inputs didn't match the amount of weights")
            raise Exception
        input_i = 0
        last_error = []
        for i in range(epochs):
            input_i += 1

            if input_i == len(inputs):
                input_i = 0
            self.pred = self.output(inputs=inputs[input_i])
            if self.pred == exp_out[input_i]:
                print("=========================================================================================\n")
                print(f"Expected output achieved in iteration {i}. Training concluded with weihgts {self.weights}.\n")
                print("=========================================================================================\n")
                break
            print(f"Pred: {self.pred}")
            error = abs(exp_out[input_i] - self.pred)
            if error in last_error:
                print("=========================================================================================\n")
                print(f"[!] Minimum error {error} achieved at iteration {i}\n")
                print("=========================================================================================\n")
                break
            last_error.append(error)
            self.adjust_weights(loss=error, learning_rate=learning_rate)
            print(f"Iteration {i}: Weights: {self.weights}, Error: {error}")
            
    def adjust_weights(self, loss, learning_rate):
        for i, w in enumerate(self.weights, start=0):
            grad = 2*(loss)*w
            print(f"Grad: {grad}")
            adjusted_w = w - (grad * learning_rate)
            self.weights[i] = adjusted_w

class Layer:
    def __init__(self, neurons, inputs, parameters, bias, size):
        self.neurons = neurons
        self.parameters = parameters
        self.bias = bias
        self.inputs = inputs
        self.size = size

        for id in self.size:
            pass

class Network:
    pass

N1 = Neuron(weights=[3, 2],bias=0.5)
N1.train(inputs=[[4, 5], [6, 10]], exp_out=[9, 18], learning_rate=0.012, epochs=100)