from models.QPR import QPRVariationalClassifier
import numpy as np
import os

#Constants
num_classes = 2
max_steps = 5000
learning_rate = 1e-3
convergence_interval = 300
num_samples = 20
batch_size = "Full Batch"
dataset = "TFIM"
if dataset == "Cluster": num_classes == 4
else: num_classes == 2

def run_all(r, n_qubits, var_ansatz_list, num_layers_list, exp_list):
    for var_ansatz in var_ansatz_list:
        for num_layers in num_layers_list:
            for exp in exp_list:
                ####
                file_path = f"results_{dataset}/{num_classes}C/{r}R/{num_qubits}Q/{var_ansatz}/{num_layers}L_5000MS_Full BatchBS_0.001LR_{convergence_interval}conv/20S/{exp}E/generalization_gap.npy"
                if os.path.exists(file_path):
                    print(f"File found: {file_path}. Next.")
                    
                ####
                else:
                    X_train = np.load(f'data/{dataset}/qubits={n_qubits}/samples={num_samples}/exp={exp}/gs.npy')
                    y_train = np.load(f'data/{dataset}/qubits={n_qubits}/samples={num_samples}/exp={exp}/label_r={r}.npy')
                    X_test = np.load(f'data/{dataset}/qubits={n_qubits}/samples=1000/exp=1/gs.npy')
                    y_test = np.load(f'data/{dataset}/qubits={n_qubits}/samples=1000/exp=1/label_r={r}.npy')

                    model = QPRVariationalClassifier(n_qubits=n_qubits, var_ansatz=var_ansatz, n_layers=num_layers, max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate,  
                                                                                    convergence_interval=convergence_interval, num_classes=num_classes, num_samples=num_samples, r=r, exp=exp)
                    model.fit(X_train, y_train)
                    g, train_acc, test_acc, mu_marg_Q1, mu_marg_Q2, mu_marg_Q3, mu_marg_mean, mu_param, mu_param_eff10, mu_param_eff100  = model.get_results(X_train, y_train, X_test, y_test)
    

var_ansatz_list = ["QCNN_not_shared"]
data_list = ["Cluster", "TFIM", "XXZ"]
num_layers_list = [1,3,5, 7,9]
exp_list = [1, 2, 3, 4, 5]
r_list = [0.25, 0.75]
num_qubits_list = [8]
for data in data_list:
    for r in r_list:
        for num_qubits in num_qubits_list:
            run_all(r, num_qubits, var_ansatz_list, num_layers_list, exp_list)