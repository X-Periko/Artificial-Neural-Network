# Artificial Neural Network

**Build, train, and understand neural networks from first principles — in pure Python.**

A zero-dependency educational library that takes you from a single artificial neuron to fully-connected multi-layer networks with real backpropagation, **without a single line of NumPy, PyTorch, or TensorFlow under the hood**. Every weight, bias, and gradient is yours to see, tweak, and reason about.

> *"The best way to understand a neural network is to build one."*

---

## ✨ Why this library?

- 🎓 **Pedagogical by design** — Readable, line-by-line implementation. No magic, no black boxes.
- 🪶 **Zero dependencies** — Pure Python standard library. Drop it into any project, any venv.
- 🔬 **From-scratch math** — Forward pass, gradient descent, and full backpropagation implemented manually.
- 🧱 **Composable architecture** — Stack `Neuron` → `Layer` → `Network` just like the textbooks describe.
- 🪞 **Transparent training** — Prints weights, biases, predictions, and errors at every step.

---

## 📦 Installation

No package manager needed. Clone, copy, or vendor — your call.

```bash
git clone https://github.com/your-user/Artificial-Neural-Network.git
cd Artificial-Neural-Network
```

That's it. No `requirements.txt`, no compiled extensions, no GPU drivers.

---

## 🚀 Quick Start

### The `Neuron` — the atom of the network

A single artificial neuron with weights, a bias, and gradient-descent training:

```python
from Network_classes import Neuron

neuron = Neuron(weights=[0.5, -0.3], bias=0.1, id=0)
output = neuron.output(inputs=[1.0, 2.0])      # forward pass
neuron.train(
    inputs=[[1.0, 2.0], [0.5, 0.5]],
    exp_out=[3.0, 1.0],
    learning_rate=0.01,
    epochs=1000,
)
```

### The `Layer` — neurons in parallel

Vectorize a column of neurons sharing the same input:

```python
from Network_classes import Layer

layer = Layer(parameters=2, size=4, id=0)      # 4 neurons, 2 inputs each
predictions = layer.predict(inputs=[1.0, 2.0]) # list of 4 outputs
```

### The `Network` — layers stacked deep

Compose an architecture and train it end-to-end with real backpropagation:

```python
from Network_classes import Network

# Architecture: 2 inputs → 3 hidden → 1 output
net = Network(arch=[[3, 2], [1, 3]])

net.train(
    inputs=[[1.0, 2.0], [0.5, 0.5], [3.0, -1.0]],
    exp_out=[3.0, 1.0, 0.5],
    learning_rate=0.01,
    epochs=200,
)

net.print_arch()                  # inspect every weight and bias
print(net.forward_pass([1.0, 2.0]))
```

---

## 🏗️ Architecture

The library mirrors the canonical three-tier model:

```
┌─────────────────────────────────────────────┐
│  Network  ── composes a list of Layers      │
│  ┌──────────────────────────────────────┐   │
│  │  Layer  ── owns a list of Neurons    │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐    │   │
│  │  │Neuron  │ │Neuron  │ │Neuron  │    │   │
│  │  └────────┘ └────────┘ └────────┘    │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

- **`Neuron`** — weights, bias, forward pass (`output`) and weight adjustment (`adjust_weights`).
- **`Layer`** — parallel column of neurons with shared input dimensionality; vectorized prediction.
- **`Network`** — stacks `Layer`s, executes forward passes, and runs full **backpropagation** across the entire stack.

---

## 🧪 The included demo

The package ships with a runnable demo baked right into `Network_classes.py`. From the project root:

```bash
python Network_classes.py
```

Watch a `Neuron` and a `Network` learn their datasets in real time — every iteration logs its weights, prediction, and error. It's the shortest path between "I have the code" and "I can see it work."

---

## 📚 API Reference

### `Neuron(weights, bias, id)`
- **`output(inputs) → float`** — Linear forward pass: `bias + Σ inputᵢ · weightᵢ`. Stores the result as `self.prediction`.
- **`train(inputs, exp_out, learning_rate, epochs)`** — Iterates the dataset cyclically. Exits early on exact match, repeated error (stagnation), or epoch limit.
- **`adjust_weights(delta, learning_rate, x)`** — Updates weights and bias by plain gradient descent.

### `Layer(parameters, size, id)`
- **`predict(inputs) → list[float]`** — Returns one output per neuron in the layer.

### `Network(arch)`
- **`arch`** — `list[list[int]]`. Each entry `[size, parameters]` describes a layer (from input to output).
- **`forward_pass(input) → list[float]`** — Runs the input through every layer in sequence.
- **`backpropagation(exp_out, learning_rate, inputs)`** — Full reverse-mode backprop, layer by layer.
- **`train(inputs, exp_out, learning_rate, epochs) → list[float]`** — Trains the network and returns the loss history.
- **`print_arch()`** — Dumps every layer, neuron, weight, and bias to stdout.

---

## ⚠️ Honest caveats

This is a **teaching library**, not a production ML framework. By design:

- 📜 **Verbose logging.** Every iteration prints to stdout — it's how you learn. Silence it when you've graduated.
- 🐢 **Pure-Python speed.** No vectorization, no BLAS, no GPU. Don't train GPT on this.
- 🧮 **Pedagogical gradients.** The single-neuron `adjust_weights` uses a simplified update rule; the `Network` class implements proper MSE backpropagation.
- 🛡️ **Sparse error handling.** A few `raise Exception` calls guard against shape mismatches — catch them in your own wrapper if you need fine-grained errors.

These aren't bugs, they're the syllabus.

---

## 🗺️ Roadmap

- Activation functions (sigmoid, ReLU, tanh) as first-class citizens
- Mini-batch training and shuffling
- Softmax + cross-entropy for classification
- A `fit()` / `predict()` scikit-learn-style façade
- Visualization helpers (decision boundaries, loss curves)

---

## 🤝 Contributing

Found a clearer way to express the math? Want to add an activation function, a new loss, or a unit test? PRs are welcome. Keep it dependency-free, keep it readable.

---

## 📄 License

MIT. Teach the world.
