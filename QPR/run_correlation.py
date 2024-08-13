from models.QPR import QPRVariationalClassifier
import numpy as np
from scipy.stats import kendalltau

#Constants
num_classes = 4
max_steps = 2000
learning_rate = 1e-3
convergence_interval = 200
num_samples = 20

def mutual_information_hist(a, b):
    num_bins = int(np.round(np.sqrt(len(a))))
    joint_hist, _, _ = np.histogram2d(a, b, bins=num_bins)
    a_hist = np.sum(joint_hist, axis=1)
    b_hist = np.sum(joint_hist, axis=0)

    # Convert histogram counts to probability values
    joint_prob = joint_hist / joint_hist.sum()
    a_prob = a_hist / a_hist.sum()
    b_prob = b_hist / b_hist.sum()

    # Make sure to replace zeros with small positive numbers to avoid log(0)
    joint_prob[joint_prob == 0] = 1e-10
    a_prob[a_prob == 0] = 1e-10
    b_prob[b_prob == 0] = 1e-10

    # Calculate the mutual information
    mi = np.sum(joint_prob * np.log2(joint_prob / (a_prob[:, None] * b_prob[None, :])))
    return mi

def run_all(r, n_qubits_list, var_ansatz_list, num_layers_list, batch_size_list, exp_list):
    g_list = np.array([])
    mu_marg_Q1_list = np.array([])
    mu_marg_Q2_list = np.array([])
    mu_marg_mean_list = np.array([])
    mu_param_list = np.array([])
    mu_param_eff_list = np.array([])
    mu_param_eff2_list = np.array([])
    mu_marg_Q1_param_list = np.array([])
    mu_marg_Q1_param_eff_list = np.array([])
    mu_marg_Q1_param_eff2_list = np.array([])
    mu_marg_Q2_param_list = np.array([])
    mu_marg_Q2_param_eff_list = np.array([])
    mu_marg_Q2_param_eff2_list = np.array([])
    mu_marg_mean_param_list = np.array([])
    mu_marg_mean_param_eff_list = np.array([])
    mu_marg_mean_param_eff2_list = np.array([])


    for n_qubits in n_qubits_list:
        for var_ansatz in var_ansatz_list:
            for num_layers in num_layers_list:
                for batch_size in batch_size_list:
                    for exp in exp_list:
                        X_train = np.load(f'data/qubits={n_qubits}/classes={num_classes}/samples={num_samples}/exp={exp}/gs.npy')
                        y_train = np.load(f'data/qubits={n_qubits}/classes={num_classes}/samples={num_samples}/exp={exp}/label_r={r}.npy')
                        X_test = np.load(f'data/qubits={n_qubits}/classes={num_classes}/samples=1000/exp=1/gs.npy')
                        y_test = np.load(f'data/qubits={n_qubits}/classes={num_classes}/samples=1000/exp=1/label_r={r}.npy')

                        model = QPRVariationalClassifier(n_qubits=n_qubits, var_ansatz=var_ansatz, n_layers=num_layers, max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate,  
                                                                                 convergence_interval=convergence_interval, num_classes=num_classes, num_samples=num_samples, r=r, exp=exp)
                        model.fit(X_train, y_train)
                        g, mu_marg_Q1, mu_marg_Q2, mu_marg_Q3, mu_marg_mean, mu_param, mu_param_eff1, mu_param_eff2  = model.get_results(X_train, y_train, X_test, y_test)

                        # List of generalization gaps
                        g_list = np.append(g_list, g)

                        # Lists of margin based complexity measures
                        mu_marg_Q1_list = np.append(mu_marg_Q1_list, mu_marg_Q1)
                        mu_marg_Q2_list = np.append(mu_marg_Q2_list, mu_marg_Q2)
                        mu_marg_mean_list = np.append(mu_marg_mean_list, mu_marg_mean)
                        
                        # Lists of parameter based complexity measures
                        mu_param_list = np.append(mu_param_list, mu_param)
                        mu_param_eff_list = np.append(mu_param_eff_list, mu_param_eff1)
                        mu_param_eff2_list = np.append(mu_param_eff2_list, mu_param_eff2)

                        # Lists of product of margin and parameter based complexity measures
                        mu_marg_Q1_param_list = np.append(mu_marg_Q1_param_list, mu_marg_Q1 * mu_param )
                        mu_marg_Q1_param_eff_list = np.append(mu_marg_Q1_param_eff_list, mu_marg_Q1 * mu_param_eff1)
                        mu_marg_Q1_param_eff2_list = np.append(mu_marg_Q1_param_eff2_list, mu_marg_Q1 * mu_param_eff2)

                        mu_marg_Q2_param_list = np.append(mu_marg_Q2_param_list, mu_marg_Q2 * mu_param )
                        mu_marg_Q2_param_eff_list = np.append(mu_marg_Q2_param_eff_list, mu_marg_Q2 * mu_param_eff1)
                        mu_marg_Q2_param_eff2_list = np.append(mu_marg_Q2_param_eff2_list, mu_marg_Q2 * mu_param_eff2)

                        mu_marg_mean_param_list = np.append(mu_marg_mean_param_list, mu_marg_mean * mu_param )
                        mu_marg_mean_param_eff_list = np.append(mu_marg_mean_param_eff_list, mu_marg_mean * mu_param_eff1)
                        mu_marg_mean_param_eff2_list = np.append(mu_marg_mean_param_eff2_list, mu_marg_mean * mu_param_eff2)
                                            
    mu_lists = np.array([mu_marg_Q1_list, mu_marg_Q2_list, mu_marg_mean_list, mu_param_list, mu_param_eff_list, mu_marg_Q1_param_list, mu_marg_Q1_param_eff_list, mu_marg_Q2_param_list, mu_marg_Q2_param_eff_list, mu_marg_mean_param_list, mu_marg_mean_param_eff_list]) 
    MI_list, Tau_list, p_value_list = [], [], []
    for mu in mu_lists:
        MI = mutual_information_hist(g_list, mu)
        tau, p_value = kendalltau(g_list, mu)
        MI_list.append(MI)
        Tau_list.append(tau)
        p_value_list.append(p_value)
    
    PATH = f'correlation_results/r={r}/'
    np.save(PATH + 'MI_list.npy', MI_list)
    np.save(PATH + 'Tau_list.npy', Tau_list)
    np.save(PATH + 'p_value_list.npy', p_value_list)

r_list = [0.0, 0.5, 1.0]

#Hyperparameters
num_qubits = [4, 8]
var_ansatz = ["QCNN_not_shared", "QCNN_shared", "SEL"]
num_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
batch_size = [5, 10]
exp_list = [1,2,3,4,5]

for r in r_list:
    run_all(r, num_qubits, var_ansatz, num_layers, batch_size, exp_list)