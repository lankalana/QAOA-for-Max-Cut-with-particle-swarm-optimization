from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

class QAOA:
    def __init__(self, p, shots, instance):
        self.simulator = AerSimulator(method='statevector')
        self.p = p
        self.shots = shots
        self.maxcut = instance
        self.max_val = self.maxcut.ComputeBitstringVal(self.maxcut.BruteForce())

    # Define the QuantumFunc which simulates the quantum circuit and computes the expected value
    def QuantumFunc(self, params):
        # Create the quantum circuit
        circuit = self.maxcut.CreateCircuit(params, self.p)
        # Simulate it
        counts = self.simulator.run(circuit, shots = self.shots).result().get_counts(0)
        # Return the expected value
        return self.maxcut.ComputeAvgFromCounts(counts)

    def ApproximationRatio(self, params, shots):
        # Create the quantum circuit
        qc = self.maxcut.CreateCircuit(params, self.p)
        # Simulate it
        result = self.simulator.run(qc, shots=shots).result().get_counts()
        expextedVal = self.maxcut.ComputeAvgFromCounts(result)
        return expextedVal / self.max_val
    
    def Top3Results(self, params, shots):
        # Create the quantum circuit
        qc = self.maxcut.CreateCircuit(params, self.p)
        # Simulate it
        result = self.simulator.run(qc, shots=shots).result().get_counts()
        top3 = sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]
        return [x[0] for x in top3]

    # Plot the results for given params. Optionally compine dubpicates
    def Plot(self, params, shots):
        # Create the quantum circuit
        qc = self.maxcut.CreateCircuit(params, self.p)
        # Simulate it
        result = self.simulator.run(qc, shots=shots).result().get_counts()
        labels = []
        counts = []
        values = []
        colors = []

        # Sort and convert the data for easier plotting
        for oneres in sorted(result.items(), key=lambda x: x[1], reverse=True):
            val = self.maxcut.ComputeBitstringVal(oneres[0])
            labels.append(oneres[0])
            counts.append(oneres[1] / shots)
            values.append(val)

            if (val == self.max_val):
                colors.append("forestgreen")
            elif (val == self.max_val - 1):
                colors.append("orange")
            else:
                colors.append("red")
        # Plot the results
        if (len(result.items()) > 32):
            print("Showing top 32 results")
        plt.bar(labels[:32], counts[:32], color=colors)
        plt.xticks(rotation=90)
        
        colors = {"Correct solution":"forestgreen", "1 off the correct solution":"orange", "Other": "red"}         
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
        plt.legend(handles, labels)
        plt.xlabel("Solution bitstring")
        plt.ylabel("Approximation ratio")