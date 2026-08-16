class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def output(self, inputs):
        self.prediction = self.bias
        for index, i in enumerate(inputs, start=0):
            self.prediction += i * self.weights[index]
        
        return self.prediction

    def train(self, inputs, exp_out, learning_rate, epochs):
        for i in range(epochs):
            self.pred = self.output(inputs=inputs)
            print(f"Pred: {self.pred}")
            error = (exp_out - self.pred)
            self.adjust_weights(loss=error, learning_rate=learning_rate)
            print(f"Iteration {i}: Weights: {self.weights}, Error: {error}")

    def adjust_weights(self, loss, learning_rate):
        for i, w in enumerate(self.weights, start=0):
            grad = 2*(loss)*w
            print(f"Grad: {grad}")
            adjusted_w = w + (grad * learning_rate)
            self.weights[i] = adjusted_w

class Layer:
    def __init__(self, neurons, inputs):
        self.neurons = neurons
        self.inputs = inputs

class Network:
    pass

N1 = Neuron(weights=[1, 4],bias=0.5)
N1.train(inputs=[3, 1], exp_out=10, learning_rate=0.012, epochs=100)