import pennylane as qml
from pennylane import numpy as np
import argparse
import os


def H(j1, j2, N):
    coeffs, ops = [], []
    for n in range(N):
        ops.append(qml.PauliZ(n))
        coeffs.append(1)
    for n in range(N - 1):
        ops.append(qml.PauliX(n) @ qml.PauliX(n+1))
        coeffs.append(-j1)
    for n in range(N - 2):
        ops.append(qml.PauliX(n) @ qml.PauliZ(n+1) @qml.PauliX(n+2))
        coeffs.append(-j2)
    H = qml.Hamiltonian(coeffs, ops)
    return qml.matrix(H)

def gs(j1, j2, N):
    eigvals, eigvecs = np.linalg.eigh(H(j1, j2, N))
    return eigvecs[:,0]

def labels(j1, j2, num_classes):
    if (j2 <= j1 - 1) and (j2 <= -j1 -1):
        y = 0
        pass_rate = 4 / 29
    elif (j2 > -j1 - 1) and (j2 <= j1 - 1):
        y = 1
        pass_rate = 4 / 15.5
    elif (j2 > j1 - 1) and (j2 <= -j1 -1):
        y = 2
        pass_rate = 4 / 15.5
    elif (j2 > j1 - 1) and (j2 > -j1 -1):
        if j2 >= 1:
            y = 0
            pass_rate = 4 / 29
        else:
            y = 3
            pass_rate = 1

    if num_classes == 2:
        if y == 0:
            y = 0
            pass_rate = 1
        else:
            y = 1
            pass_rate = 29 / 35
    return y, pass_rate
        

def gen_data(num_qubits, num_classes, num_samples, exp):
    j1j2_list = []
    gs_list = []
    y_list = []

    directory = f'Cluster/qubits={num_qubits}/classes={num_classes}/samples={num_samples}/exp={exp}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    sample_count = 0
    while sample_count < num_samples:
        j1, j2 = 8 * (np.random.random() - 0.5), 8 * (np.random.random() - 0.5)
        y, pass_rate = labels(j1, j2, num_classes)
        
        rand = np.random.random()
        if rand < pass_rate:
            ground_state = gs(j1, j2, num_qubits)
            j1j2_list.append([j1, j2])
            gs_list.append(ground_state)
            y_list.append(y)
            sample_count += 1
    
    with open(os.path.join(directory, 'gs_j1j2.txt'), 'w') as f:
        f.write(str(j1j2_list))
    
    gs_list = np.array(gs_list)
    np.save(os.path.join(directory, 'gs.npy'), gs_list)

    y_list = np.array(y_list)
    np.save(os.path.join(directory, 'label_r=0.0.npy'), y_list)
    
    for noise_level in [0.0, 0.25, 0.5, 0.75, 1.0]:
        label_file = os.path.join(directory, f'label_r={noise_level:.2f}.npy')
        if not os.path.exists(label_file):
            noisy_labels = y_list.copy()
            random_indices = np.random.choice(num_samples, int(noise_level * num_samples), replace=False)
            noisy_labels[random_indices] = np.random.randint(0, num_classes, size=len(random_indices))
            np.save(label_file, noisy_labels)

    
def gen_data_all(num_qubits_list, num_classes_list, num_samples_list, exp_list):
    for num_qubits in num_qubits_list:
        for num_classes in num_classes_list:
            for num_samples in num_samples_list:
                for exp in exp_list:
                    path = f'Cluster/qubits={num_qubits}/classes={num_classes}/samples={num_samples}/exp={exp}/'
                    os.makedirs(path, exist_ok=True)
                    gen_data(num_qubits, num_classes, num_samples, exp)
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits_list", nargs="+", type=int, default=4)  # 4, 8 ,16, 32
    parser.add_argument("--num_classes_list", nargs="+", type=int, default=2) # 2, 3, 4
    parser.add_argument("--num_samples_list", nargs="+", type=int, default=5) # 5, 10, 15, 20, 25
    parser.add_argument("--exp_list", nargs="+", type=int, default=range(1,51))         # 1, 2, 3, 4, 5
    args, _ = parser.parse_known_args()
    gen_data_all(**vars(args))