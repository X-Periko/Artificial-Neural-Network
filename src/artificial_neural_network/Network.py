import random as r
last_n_id = -1
last_l_id = -1

class Neuron:
    def __init__(self, weights, bias,id):
        self.weights = weights
        self.bias = bias
        self.id = id

    def output(self, inputs):
        self.prediction = self.bias
        for index, i in enumerate(inputs, start=0):
            self.prediction += i * self.weights[index]
        
        return self.prediction
            
    def adjust_weights(self, delta, learning_rate, x):
        for i, w in enumerate(self.weights):
            grad = delta * x[i]
            self.weights[i] = w - learning_rate * grad
        self.bias = self.bias - learning_rate * delta

class Layer:
    def __init__(self, parameters:int, size:int, id):
        self.neurons = []
        self.parameters = parameters
        self.size = size
        self.id = id

        global last_n_id
        for _ in range(self.size):
            weights = [r.randint(-10, 10) for _ in range(self.parameters)]
            last_n_id += 1
            N = Neuron(weights=weights, bias=r.randint(-10, 10), id=last_n_id)
            self.neurons.append(N)

    def predict(self, inputs:list) -> list:
        prediction = [N.output(inputs) for N in self.neurons]
        return prediction

class Network:
    def __init__(self, arch:list[list]):
        self.arch = arch
        self.layers = []

        global last_l_id
        for index, i in enumerate(arch, start=0):
            last_l_id +=1
            L = Layer(size=i[0], parameters=i[1], id=last_l_id)
            self.layers.append(L)

    def print_arch(self):
        for l in self.layers:
            print(f"\nLayer {l.id}: {l.size} Neurons with {l.parameters} parameters")
            for n in l.neurons:
                print(f"    - Neuron {n.id} with parameters {n.weights} and bias {n.bias}")
        print("==========================================================================\n\n")

    def forward_pass(self, input:list) -> list:
        pred = input
        for l in self.layers:
            pred = l.predict(pred)
        return pred

    def backpropagation(self, exp_out, learning_rate, inputs):
        activations = [inputs]
        current = inputs
        for l in self.layers:
            current = l.predict(current)
            activations.append(current)

        output = activations[-1]
        deltas = [2 * (o - e) for o, e in zip(output, exp_out)]

        for l_index in reversed(range(len(self.layers))):
            layer = self.layers[l_index]
            layer_input = activations[l_index]

            for n, delta in zip(layer.neurons, deltas):
                n.adjust_weights(delta=delta, learning_rate=learning_rate, x=layer_input)

            if l_index > 0:
                prev_layer = self.layers[l_index - 1]
                new_deltas = []
                for i, prev_neuron in enumerate(prev_layer.neurons):
                    s = sum(deltas[j] * layer.neurons[j].weights[i] for j in range(len(layer.neurons)))
                    new_deltas.append(s)
                deltas = new_deltas

    def train(self, inputs:list[list], exp_out:list, learning_rate, epochs:int):
        input_i = 0
        needed_weights = len(inputs[0])
        loss_history = []
        for input in inputs:
            if needed_weights != len(input):
                print("[!] Inputs with different lenghts were provided")
                raise Exception
        if self.layers[0].parameters != needed_weights:
            print("[!] Inputs didn't match weights")
        for i in range(epochs):
            input_i += 1
            if input_i == len(inputs):
                input_i = 0
            pred = self.forward_pass(inputs[input_i])
            # calculamos el loss real (error cuadrático) solo para poder verlo, no para backprop
            loss = sum((p - e)**2 for p, e in zip(pred, exp_out[input_i]))
            if loss in loss_history:
                            print("=========================================================================================\n")
                            print(f"[!] Minimum error {loss} achieved at iteration {i}\n")
                            print("=========================================================================================\n")
                            break
            loss_history.append(loss)
            self.backpropagation(exp_out=exp_out[input_i], learning_rate=learning_rate, inputs=inputs[input_i])
            if i % 10== 0:
                self.print_arch()
        return loss_history