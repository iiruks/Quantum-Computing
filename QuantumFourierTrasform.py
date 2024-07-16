import numpy as np

def qft(state_vector):
    """Apply Quantum Fourier Transform to a given state vector."""
    n = len(state_vector)
    qft_matrix = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            qft_matrix[i, j] = np.exp(2j * np.pi * i * j / n) / np.sqrt(n)
    return np.dot(qft_matrix, state_vector)

def main():
    # Define the number of qubits
    n_qubits = 3
    # Define the initial state vector (e.g., |0⟩ state)
    state_vector = np.zeros(2**n_qubits, dtype=complex)
    state_vector[0] = 1.0  # Starting in the |000⟩ state
    
    # Initialize the state vector with a sample state (e.g., |000⟩ + |101⟩)
    state_vector[0] = 1/np.sqrt(2)
    state_vector[5] = 1/np.sqrt(2)
    
    print("Initial state vector:")
    print(state_vector)
    
    # Apply QFT
    transformed_state = qft(state_vector)
    
    print("\nState vector after QFT:")
    print(transformed_state)

if __name__ == "__main__":
    main()
