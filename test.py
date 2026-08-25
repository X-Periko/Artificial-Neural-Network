from Network_classes import *
import matplotlib.pyplot as plt
import networkx as nx

Net = Network(arch=[
    [16, 2],
    [8, 16],
    [4, 8],
    [2, 4],
    [1, 2]
])
Net.print_arch()

inputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 1],
    [1, 2],
    [3, 0],
    [0, 3],
    [2, 2],
    [-1, 1],
]

exp_out = [
    [-1],
    [1],
    [2],
    [4],
    [6],
    [7],
    [5],
    [8],
    [9],
    [0],
]

loss_history = Net.train(inputs=inputs,exp_out=exp_out,learning_rate=0.000000000001,epochs=100000)
print(loss_history[-1])
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolución del error durante el entrenamiento")
plt.savefig("loss.png")
print("Plot saved to loss.png")

def plot_architecture(network):
    G = nx.DiGraph()
    pos = {} 
    n_inputs = network.layers[0].parameters
    input_nodes = [f"in_{i}" for i in range(n_inputs)]
    for i, node in enumerate(input_nodes):
        G.add_node(node)
        pos[node] = (0, -i + n_inputs / 2)

    prev_nodes = input_nodes

    for layer_idx, layer in enumerate(network.layers):
        current_nodes = [f"L{layer_idx}_N{n.id}" for n in layer.neurons]
        for i, node in enumerate(current_nodes):
            G.add_node(node)
            pos[node] = (layer_idx + 1, -i + layer.size / 2)
        for prev_node in prev_nodes:
            for curr_node in current_nodes:
                G.add_edge(prev_node, curr_node)

        prev_nodes = current_nodes

    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="lightblue",
        node_size=1200,
        font_size=7,
        arrows=True,
        arrowsize=10
    )
    plt.title("Arquitectura de la red")
    plt.axis("off")
    plt.savefig("graph.png")

plot_architecture(Net)
print("Graph saved to graph.png")