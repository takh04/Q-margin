import pennylane as qml
from pennylane import numpy as np
import argparse
import os


def H(J, g, N):
    coeffs, ops = [], []
    # Nearest-neighbor ZZ interaction
    for n in range(N - 1):
        ops.append(qml.PauliZ(n) @ qml.PauliZ(n+1))
        coeffs.append(-J)
    # Transverse field term (g term)
    for n in range(N):
        ops.append(qml.PauliX(n))
        coeffs.append(-J * g)
    H = qml.Hamiltonian(coeffs, ops)
    return qml.matrix(H)

def gs(J, g, N):
    eigvals, eigvecs = np.linalg.eigh(H(J, g, N))
    return eigvecs[:,0]

def labels(J, g):
    # Binary classification for TFIM in ordered phase (|g| < 1):
    # - Ferromagnetic phase: J > 0
    # - Antiferromagnetic phase: J < 0
    if J > 0:
        y = 0  # Ferromagnetic phase
    else:
        y = 1  # Antiferromagnetic phase
    return y

def gen_data(num_qubits, num_samples, exp):
    Jg_list = []
    gs_list = []
    y_list = []

    directory = f'TFIM/qubits={num_qubits}/samples={num_samples}/exp={exp}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    sample_count = 0
    while sample_count < num_samples:
        # Sample J from [-1, 1] range (excluding 0)
        J = 2 * (np.random.random() - 0.5)
        if abs(J) < 0.1:  # Avoid J too close to 0
            continue
            
        # Sample g from [-0.9, 0.9] range to ensure ordered phase (|g| < 1)
        g = 1.8 * (np.random.random() - 0.5)
        
        y = labels(J, g)
        
        ground_state = gs(J, g, num_qubits)
        Jg_list.append([J, g])
        gs_list.append(ground_state)
        y_list.append(y)
        sample_count += 1
    
    with open(os.path.join(directory, 'gs_Jg.txt'), 'w') as f:
        f.write(str(Jg_list))
    
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
                path = f'TFIM/qubits={num_qubits}/samples={num_samples}/exp={exp}/'
                os.makedirs(path, exist_ok=True)
                gen_data(num_qubits, num_samples, exp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits_list", nargs="+", type=int, default=4)  # 4, 8, 16, 32
    parser.add_argument("--num_samples_list", nargs="+", type=int, default=5)  # 5, 10, 15, 20, 25
    parser.add_argument("--exp_list", nargs="+", type=int, default=range(1,51))  # 1, 2, 3, 4, 5
    args, _ = parser.parse_known_args()
    gen_data_all(**vars(args)) 