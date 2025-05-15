import numpy as np
from scipy.stats import kendalltau
import os

# Constants
convergence_interval = 300
dataset = "XXZ"
if dataset == "Cluster":
    num_classes = 4
else:
    num_classes = 2

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

def get_correlation(g_lists, test_acc_lists, mu_lists):
    MI_g_mu_list, Tau_g_mu_list, p_value_g_mu_list, MI_test_mu_list, Tau_test_mu_list, p_value_test_mu_list = [], [], [], [], [], []
    for mu_list in mu_lists:
        MI_g_mu = mutual_information_hist(g_lists, mu_list)
        Tau_g_mu, p_value_g_mu = kendalltau(g_lists, mu_list, nan_policy='omit')
        MI_g_mu_list.append(MI_g_mu)
        Tau_g_mu_list.append(Tau_g_mu)
        p_value_g_mu_list.append(p_value_g_mu)

        MI_test_mu = mutual_information_hist(test_acc_lists, mu_list)
        Tau_test_mu, p_value_test_mu = kendalltau(test_acc_lists, mu_list, nan_policy='omit')
        MI_test_mu_list.append(MI_test_mu)
        Tau_test_mu_list.append(Tau_test_mu)
        p_value_test_mu_list.append(p_value_test_mu)
    return MI_g_mu_list, Tau_g_mu_list, p_value_g_mu_list, MI_test_mu_list, Tau_test_mu_list, p_value_test_mu_list

def calculate_correlations(r, n_qubits, var_ansatz_list, num_layers_list, exp_list):
    g_list = np.array([])
    train_acc_list = np.array([])
    test_acc_list = np.array([])

    mu_marg_Q1_list = np.array([])
    mu_marg_Q2_list = np.array([])
    mu_marg_mean_list = np.array([])
    mu_param_list = np.array([])
    mu_param_eff10_list = np.array([])
    mu_param_eff100_list = np.array([])

    for var_ansatz in var_ansatz_list:
        for num_layers in num_layers_list:
            for exp in exp_list:
                base_path = f"results_{dataset}/{num_classes}C/{r}R/{n_qubits}Q/{var_ansatz}/{num_layers}L_5000MS_Full BatchBS_0.001LR_{convergence_interval}conv/20S/{exp}E"
                
                # Load all metrics
                g = np.load(f"{base_path}/generalization_gap.npy")
                train_acc = np.load(f"{base_path}/train_acc.npy")
                test_acc = np.load(f"{base_path}/test_acc.npy")
                mu_marg_Q1 = np.load(f"{base_path}/mu_marg_Q1.npy")
                mu_marg_Q2 = np.load(f"{base_path}/mu_marg_Q2.npy")
                mu_marg_mean = np.load(f"{base_path}/mu_marg_mean.npy")
                mu_param = np.load(f"{base_path}/mu_param.npy")
                mu_param_eff10 = np.load(f"{base_path}/mu_param_eff10.npy")
                mu_param_eff100 = np.load(f"{base_path}/mu_param_eff100.npy")

                # Append to lists
                g_list = np.append(g_list, g)
                train_acc_list = np.append(train_acc_list, train_acc)
                test_acc_list = np.append(test_acc_list, test_acc)
                mu_marg_Q1_list = np.append(mu_marg_Q1_list, mu_marg_Q1)
                mu_marg_Q2_list = np.append(mu_marg_Q2_list, mu_marg_Q2)
                mu_marg_mean_list = np.append(mu_marg_mean_list, mu_marg_mean)
                mu_param_list = np.append(mu_param_list, mu_param)
                mu_param_eff10_list = np.append(mu_param_eff10_list, mu_param_eff10)
                mu_param_eff100_list = np.append(mu_param_eff100_list, mu_param_eff100)

    # Calculate correlations
    mu_lists = np.array([mu_marg_Q1_list, mu_marg_Q2_list, mu_marg_mean_list, mu_param_list, mu_param_eff10_list, mu_param_eff100_list])
    MI_g_mu_list, Tau_g_mu_list, p_value_g_mu_list, MI_test_mu_list, Tau_test_mu_list, p_value_test_mu_list = get_correlation(g_list, test_acc_list, mu_lists)
    
    # Save results
    PATH = f'results_correlation/{dataset}/r={r}/num_qubits={n_qubits}/'
    if not os.path.exists(PATH + 'others/'):
        os.makedirs(PATH + 'others/')
    
    # Save lists
    np.save(PATH + 'others/g_list.npy', g_list)
    np.save(PATH + 'others/train_acc_list.npy', train_acc_list)
    np.save(PATH + 'others/test_acc_list.npy', test_acc_list)
    np.save(PATH + 'others/mu_marg_Q1_list.npy', mu_marg_Q1_list)
    np.save(PATH + 'others/mu_marg_Q2_list.npy', mu_marg_Q2_list)
    np.save(PATH + 'others/mu_marg_mean_list.npy', mu_marg_mean_list)
    np.save(PATH + 'others/mu_param_list.npy', mu_param_list)
    np.save(PATH + 'others/mu_param_eff10_list.npy', mu_param_eff10_list)
    np.save(PATH + 'others/mu_param_eff100_list.npy', mu_param_eff100_list)

    # Save correlation results
    np.save(PATH + 'MI_g_mu_list.npy', MI_g_mu_list)
    np.save(PATH + 'Tau_g_mu_list.npy', Tau_g_mu_list)
    np.save(PATH + 'p_value_g_mu_list.npy', p_value_g_mu_list)
    np.save(PATH + 'MI_test_mu_list.npy', MI_test_mu_list)
    np.save(PATH + 'Tau_test_mu_list.npy', Tau_test_mu_list)
    np.save(PATH + 'p_value_test_mu_list.npy', p_value_test_mu_list)

    print(f"Results saved to {PATH}")

if __name__ == "__main__":
    var_ansatz_list = ["QCNN_not_shared", "QCNN_shared", "SEL"]
    num_layers_list = [1,3,5,7,9]
    exp_list = [1, 2, 3, 4, 5]
    r_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_qubits_list = [10]

    for r in r_list:
        for num_qubits in num_qubits_list:
            calculate_correlations(r, num_qubits, var_ansatz_list, num_layers_list, exp_list) 