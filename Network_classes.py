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
            if input_i == len(inputs):
                input_i = 0
            self.pred = self.output(inputs=inputs[input_i])
            if self.pred == exp_out[input_i]:
                print("=========================================================================================\n")
                print(f"Expected output achieved in iteration {i}. Training concluded with weihgts {self.weights}.\n")
                print("=========================================================================================\n")
                break
            print(f"Pred: {self.pred}")
            signed_error = exp_out[input_i] - self.pred
            error = abs(signed_error)
            if error in last_error:
                print("=========================================================================================\n")
                print(f"[!] Minimum error {error} achieved at iteration {i}\n")
                print("=========================================================================================\n")
                break
            last_error.append(error)
            self.adjust_weights(loss=signed_error, learning_rate=learning_rate, x = inputs[input_i])
            print(f"Iteration {i}: Weights: {self.weights}, Error: {error}")
            input_i += 1
            
    def adjust_weights(self, loss, learning_rate,x):
        for i, w in enumerate(self.weights, start=0):
            grad = -2*(loss)*x[i]
            print(f"Grad: {grad}")
            adjusted_w = w - (grad * learning_rate)
            self.weights[i] = adjusted_w

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
            print(f"Neuron with id {last_n_id} created with weights {weights} and bias {N.bias}")

    def predict(self, inputs:list):
        prediction = [N.output(inputs) for N in self.neurons]
        return prediction

class Network:
    def __init__(self, arch:list[list]):
        self.arch = arch
        self.layers = []

        for i in arch:
            last_l_id +=1
            L = Layer(parameters=i[1], size=i[0], id=last_l_id)
            self.layers.append()

#N1 = Neuron(weights=[3, 2],bias=0.5)
#N1.train(inputs=[[4, 5], [6, 10]], exp_out=[9, 18], learning_rate=0.012, epochs=1000)

#L1 = Layer(parameters=3, size=4)
#print(L1.predict(inputs=[1,3,7]))