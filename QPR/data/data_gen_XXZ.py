import pennylane as qml
from pennylane import numpy as np
import argparse
import os


def H(J, Delta, N):
    coeffs, ops = [], []
    # Nearest-neighbor XXZ interaction
    for n in range(N - 1):
        # XX term
        ops.append(qml.PauliX(n) @ qml.PauliX(n+1))
        coeffs.append(J)
        # YY term
        ops.append(qml.PauliY(n) @ qml.PauliY(n+1))
        coeffs.append(J)
        # ZZ term with anisotropy
        ops.append(qml.PauliZ(n) @ qml.PauliZ(n+1))
        coeffs.append(J * Delta)
    H = qml.Hamiltonian(coeffs, ops)
    return qml.matrix(H)

def gs(J, Delta, N):
    eigvals, eigvecs = np.linalg.eigh(H(J, Delta, N))
    return eigvecs[:,0]

def labels(Delta):
    # Binary classification for XXZ model with J=1:
    # - Ferromagnetic phase: Delta < -1
    # - Antiferromagnetic phase: Delta > 1
    if Delta < -1:
        y = 0  # Ferromagnetic phase
    else:
        y = 1  # Antiferromagnetic phase
    return y

def gen_data(num_qubits, num_samples, exp):
    Delta_list = []
    gs_list = []
    y_list = []

    directory = f'XXZ/qubits={num_qubits}/samples={num_samples}/exp={exp}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Ensure even number of samples for balanced dataset
    if num_samples % 2 != 0:
        num_samples += 1
    
    # Generate samples for each class
    samples_per_class = num_samples // 2
    
    # Generate ferromagnetic samples (Delta < -1)
    for _ in range(samples_per_class):
        # Sample Delta from [-2, -1) range
        Delta = -2 + np.random.random()
        J = 1  # Fixed coupling constant
        
        ground_state = gs(J, Delta, num_qubits)
        Delta_list.append([J, Delta])
        gs_list.append(ground_state)
        y_list.append(0)  # Ferromagnetic phase
    
    # Generate antiferromagnetic samples (Delta > 1)
    for _ in range(samples_per_class):
        # Sample Delta from (1, 2] range
        Delta = 1 + np.random.random()
        J = 1  # Fixed coupling constant
        
        ground_state = gs(J, Delta, num_qubits)
        Delta_list.append([J, Delta])
        gs_list.append(ground_state)
        y_list.append(1)  # Antiferromagnetic phase
    
    # Shuffle the data
    indices = np.random.permutation(num_samples)
    Delta_list = [Delta_list[i] for i in indices]
    gs_list = [gs_list[i] for i in indices]
    y_list = [y_list[i] for i in indices]
    
    with open(os.path.join(directory, 'gs_JDelta.txt'), 'w') as f:
        f.write(str(Delta_list))
    
    gs_list = np.array(gs_list)
    np.save(os.path.join(directory, 'gs.npy'), gs_list)

    y_list = np.array(y_list)
    np.save(os.path.join(directory, 'label_r=0.0.npy'), y_list)
    
    for noise_level in [0.0, 0.5, 1.0]:
        noisy_labels = y_list.copy()
        random_indices = np.random.choice(num_samples, int(noise_level * num_samples), replace=False)
        noisy_labels[random_indices] = np.random.randint(0, 2, size=len(random_indices))
        np.save(os.path.join(directory, f'label_r={noise_level:.1f}.npy'), noisy_labels)

def gen_data_all(num_qubits_list, num_samples_list, exp_list):
    for num_qubits in num_qubits_list:
        for num_samples in num_samples_list:
            for exp in exp_list:
                path = f'XXZ/qubits={num_qubits}/samples={num_samples}/exp={exp}/'
                os.makedirs(path, exist_ok=True)
                gen_data(num_qubits, num_samples, exp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits_list", nargs="+", type=int, default=4)  # 4, 8, 16, 32
    parser.add_argument("--num_samples_list", nargs="+", type=int, default=5)  # 5, 10, 15, 20, 25
    parser.add_argument("--exp_list", nargs="+", type=int, default=range(1,51))  # 1, 2, 3, 4, 5
    args, _ = parser.parse_known_args()
    gen_data_all(**vars(args)) 