from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

class QAOA:
    def __init__(self, p, shots, instance):
        self.simulator = AerSimulator(method='statevector')
        self.p = p
        self.shots = shots
        self.maxcut = instance

    # Define the QuantumFunc which simulates the quantum circuit and computes the expected value
    def QuantumFunc(self, params):
        # Create the quantum circuit
        circuit = self.maxcut.CreateCircuit(params, self.p)
        # Simulate it
        counts = self.simulator.run(circuit, shots = self.shots).result().get_counts(0)
        # Return the expected value
        return self.maxcut.ComputeAvgFromCounts(counts)

    # Plot the results for given params. Optionally compine dubpicates
    def Plot(self, params, shots, skipInverse=False):
        # Create the quantum circuit
        qc = self.maxcut.CreateCircuit(params, self.p)
        # Simulate it
        result = self.simulator.run(qc, shots=shots).result().get_counts()
        labels = []
        counts = []
        values = []
        max_val = 0
        inverses = {}
        # Sort and convert the data for easier plotting
        if (len(result.items()) > 32):
            print("Showing top 32 results")
        for oneres in sorted(result.items(), key=lambda x: x[1], reverse=True)[:32]:
            val = self.maxcut.ComputeBitstringVal(oneres[0])
            if (val > max_val):
                max_val = val
            invVal = ''.join('1' if x == '0' else '0' for x in oneres[0])
            # Combine the counts for inverses, e.g. 0011 and 1100
            if (skipInverse == True):
                existing = inverses.get(invVal)
                if (existing != None):
                    counts[existing] += int(oneres[1])
                    continue
                inverses[oneres[0]] = len(labels)
            labels.append(oneres[0])
            counts.append(oneres[1] / shots)
            values.append(val)
        # Print some key values
        print("Approximation ratio: {ar:.2f}".format(ar=self.maxcut.ComputeAvgFromCounts(result) / max_val))
        # Plot the results
        plt.bar(labels, counts, color=[("green" if x == max_val else ("orange" if x == max_val - 1 else "red")) for x in values])
        plt.xticks(rotation=90)
        plt.show()