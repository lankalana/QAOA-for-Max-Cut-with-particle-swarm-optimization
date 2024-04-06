from qiskit import QuantumCircuit
import networkx as nx
import numpy as np

### Simple graphs ###
def TriangleGraph():
    return CircleGraph(3)

def SquareGraph():
    return CircleGraph(4)

def HalvedSquareGraph():
    return MaxCut(4, [(0, 1, 1.), (1, 3, 1.), (0, 2, 1.), (2, 3, 1), (0, 3, 1.)])

### More complex graphs ###
# Creates a graph where all nodes are connected to the next and the previous. Uses random weights in range [0,10] if randomWeights is True
def CircleGraph(nodeCnt, randomWeights=False, seed=None):
    edges = []
    gen = np.random.default_rng(seed=seed)
    for i in range(nodeCnt):
        edges.append((i, (i + 1) % nodeCnt, gen.random() * 10. if randomWeights == True else 1.))
    return MaxCut(nodeCnt, edges)

# Creates a graph where all nodes are connected to all the other ones. Uses random weights in range [0,10] if randomWeights is True
def FullGraph(nodeCnt, randomWeights=False, seed=None):
    edges = []
    gen = np.random.default_rng(seed=seed)
    for i in range(nodeCnt):
        for j in range(i + 1, nodeCnt):
            edges.append((i, j, gen.random() * 10. if randomWeights == True else 1.))
    return MaxCut(nodeCnt, edges)

# Creates a graph where all nodes are connected to all the other ones. Uses random weights in range [0,10] if randomWeights is True
def RandomConnectedGraph(nodeCnt, randomWeights=False, seed=None):
    edges = []
    gen = np.random.default_rng(seed=seed)
    for i in range(nodeCnt):
        for j in range(i + 1, nodeCnt):
            if (gen.choice([True, False])):
                edges.append((i, j, gen.random() * 10. if randomWeights == True else 1.))
    return MaxCut(nodeCnt, edges)

### Max-Cut class ###
class MaxCut:
    # Initialize with the given number of nodes and edges between them. 
    def __init__(self, nodeCnt, edges):
        self.nodeCnt = nodeCnt
        self.edges = edges
        # Prepare the range [0, nodeCnt[ to be used with for-loops
        self.rn = range(self.nodeCnt)
        # Prepare a quantum circuit to apply a Hadamard gate to all qubits
        self.hadCircuit = QuantumCircuit(self.nodeCnt)
        for i in self.rn:
            self.hadCircuit.h(i)

    # Show the graph using networkx and matplotlib. Optionally color the nodes according to the given bitstring
    def PrintGraph(self, bitstring=None):
        if (bitstring == None):
            bitstring = '0' * self.nodeCnt
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, self.nodeCnt, 1))
        G.add_weighted_edges_from(self.edges)
        nodeColors = []
        for i in range(self.nodeCnt):
            nodeColors.append('r' if bitstring[i] == '1' else 'b')
        edgeColors = []
        for a, b in G.edges():
            edgeColors.append('0' if (bitstring[a] == bitstring[b]) else 'r')
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, node_color=nodeColors, edge_color=edgeColors, node_size=600, alpha=0.8, pos=pos)
        if any([x[2] != 1 for x in self.edges]):
            nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, "weight"))

    # Create the quantum circuit representation of the graph
    def CreateCircuit(self, params, p):
        # Use the pre-generated initalization circuit
        circuit = self.hadCircuit.copy()
        # Repeat the cost and mixer layers p times
        for level in range(p):
            # Cost layer
            for pair in self.edges:
                circuit.rzz(-2*pair[2]*params[2*level], pair[0], pair[1])
            # Mixer circuit
            for i in self.rn:
                circuit.rx(2*params[2*level + 1], i)
        circuit.measure_all()
        return circuit
    
    # Computes the cost function for a given bitstring
    def ComputeBitstringVal(self, bitstring):
        val = 0
        for a, b, w in self.edges:
            # Check if there is no connection between a and b
            if (bitstring[a] != bitstring[b]):
                val += w
        return val

    # Computes the expected value from the distribution returned from running the quantum circuit
    def ComputeAvgFromCounts(self, counts):
        sum = 0
        totalCnt = 0
        for bits, cnt in counts.items():
            # Same as ComputeBitstringVal, gives a small speedup when the function call is not needed
            for a, b, w in self.edges:
                if (bits[a] != bits[b]):
                    sum += w * cnt
            totalCnt += cnt
        return sum / totalCnt