import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def create_oracle(nqubits, marked_state):
    """Creates an oracle matrix marking the state 'marked_state'."""
    oracle = np.identity(2**nqubits, dtype=complex)
    index = int(marked_state, 2)
    oracle[index, index] = -1
    return oracle

def create_diffuser(nqubits):
    """Creates a diffuser matrix."""
    diffuser = 2 * np.ones((2**nqubits, 2**nqubits), dtype=complex) / 2**nqubits - np.identity(2**nqubits, dtype=complex)
    return diffuser

def apply_gate(state_vector, gate):
    """Applies a quantum gate to a state vector."""
    return np.dot(gate, state_vector)

def initialize_state(nqubits):
    """Initializes the state to an equal superposition."""
    state_vector = np.ones(2**nqubits, dtype=complex) / np.sqrt(2**nqubits)
    return state_vector

def measure(state_vector, nshots=1024):
    """Simulates measurements of the state vector."""
    probabilities = np.abs(state_vector)**2
    outcomes = [bin(i)[2:].zfill(int(np.log2(len(state_vector)))) for i in range(len(state_vector))]
    measurements = np.random.choice(outcomes, size=nshots, p=probabilities)
    counts = dict(Counter(measurements))
    return counts

def grover_search(nqubits, marked_state, iterations=1):
    """Performs Grover's search algorithm."""
    oracle = create_oracle(nqubits, marked_state)
    diffuser = create_diffuser(nqubits)
    state_vector = initialize_state(nqubits)
    
    for _ in range(iterations):
        state_vector = apply_gate(state_vector, oracle)
        state_vector = apply_gate(state_vector, diffuser)
    
    return state_vector

def main():
    nqubits = 3
    marked_state = '101'
    iterations = 1
    
    state_vector = grover_search(nqubits, marked_state, iterations)
    
    # Simulate measurements
    counts = measure(state_vector)
    
    print("Measurement results:", counts)
    
    # Plot the results
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('State')
    plt.ylabel('Counts')
    plt.show()

if __name__ == "__main__":
    main()
