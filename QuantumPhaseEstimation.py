import numpy as np
import matplotlib.pyplot as plt

def hadamard_transform(n):
    """Create an n-qubit Hadamard gate."""
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H_n = H
    for _ in range(n - 1):
        H_n = np.kron(H_n, H)
    return H_n

def controlled_unitary(U, n_ancilla, control):
    """Create a controlled unitary gate."""
    dim = 2**(n_ancilla + U.shape[0])
    CU = np.identity(dim, dtype=complex)
    for i in range(2**n_ancilla):
        control_bit = (i >> control) & 1
        if control_bit == 1:
            for j in range(U.shape[0]):
                CU[i * U.shape[0] + j, i * U.shape[0] + j] = U[j, j]
    return CU

def qpe(U, n_eigen, n_ancilla):
    """Quantum Phase Estimation algorithm."""
    # Initial state
    eigen_state = np.zeros(2**n_eigen, dtype=complex)
    eigen_state[0] = 1.0
    state = np.kron(np.ones(2**n_ancilla) / np.sqrt(2**n_ancilla), eigen_state)
    
    # Apply Hadamard to ancilla qubits
    H_ancilla = hadamard_transform(n_ancilla)
    state = np.dot(np.kron(H_ancilla, np.identity(2**n_eigen)), state)
    
    # Apply controlled-U operations
    for i in range(n_ancilla):
        CU = controlled_unitary(np.linalg.matrix_power(U, 2**i), n_ancilla, i)
        state = np.dot(CU, state)
    
    # Apply inverse QFT
    state = np.dot(np.kron(np.linalg.inv(hadamard_transform(n_ancilla)), np.identity(2**n_eigen)), state)
    
    # Measure ancilla qubits
    probabilities = np.abs(state)**2
    measurement = np.random.choice(range(len(probabilities)), p=probabilities)
    phase_estimate = measurement / (2**n_ancilla)
    
    return phase_estimate, state

def main():
    # Define the unitary matrix (example: a 2x2 matrix with a known eigenvalue e^(2*pi*i*0.25))
    theta = 0.25
    U = np.array([[1, 0], [0, np.exp(2j * np.pi * theta)]])
    
    n_eigen = 1  # Number of qubits for eigenstate
    n_ancilla = 3  # Number of ancilla qubits
    
    phase, state = qpe(U, n_eigen, n_ancilla)
    
    print("Estimated phase:", phase)
    print("Actual phase:", theta)
    
    # Plot the state vector probabilities
    probabilities = np.abs(state)**2
    plt.bar(range(len(probabilities)), probabilities)
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.show()

if __name__ == "__main__":
    main()
