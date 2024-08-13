from models.QPR import QPRVariationalClassifier
import numpy as np
import argparse

def mutual_information_hist(a, b, num_bins):
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

def run_all(n_qubits_list, var_ansatz_list, num_layers_list, max_steps_list, batch_size_list, learning_rate_list, convergence_interval_list, num_classes_list, num_samples_list, r_list, exp_list):
    g_list = np.array([])
    mu_marg_10_list = np.array([])
    mu_marg_Q1_list = np.array([])
    mu_marg_Q2_list = np.array([])
    mu_marg_Q3_list = np.array([])
    mu_marg_mean_list = np.array([])
    mu_param_list = np.array([])
    mu_param_eff1_list = np.array([])
    mu_param_eff10_list = np.array([])
    mu_param_eff100_list = np.array([])
    mu_marg_10_param_list = np.array([])
    mu_marg_10_param_eff1_list = np.array([])
    mu_marg_10_param_eff10_list = np.array([])
    mu_marg_10_param_eff100_list = np.array([])
    mu_marg_Q1_param_list = np.array([])
    mu_marg_Q1_param_eff1_list = np.array([])
    mu_marg_Q1_param_eff10_list = np.array([])
    mu_marg_Q1_param_eff100_list = np.array([])
    mu_marg_Q2_param_list = np.array([])
    mu_marg_Q2_param_eff1_list = np.array([])
    mu_marg_Q2_param_eff10_list = np.array([])
    mu_marg_Q2_param_eff100_list = np.array([])
    mu_marg_Q3_param_list = np.array([])
    mu_marg_Q3_param_eff1_list = np.array([])
    mu_marg_Q3_param_eff10_list = np.array([])
    mu_marg_Q3_param_eff100_list = np.array([])
    mu_marg_mean_param_list = np.array([])
    mu_marg_mean_param_eff1_list = np.array([])
    mu_marg_mean_param_eff10_list = np.array([])
    mu_marg_mean_param_eff100_list = np.array([])

    for num_classes in num_classes_list:
        for r in r_list:
            for n_qubits in n_qubits_list:
                for var_ansatz in var_ansatz_list:
                    for num_layers in num_layers_list:
                        for max_steps in max_steps_list: 
                            for batch_size in batch_size_list:
                                for learning_rate in learning_rate_list: 
                                    for convergence_interval in convergence_interval_list: 
                                        for num_samples in num_samples_list: 
                                            for exp in exp_list: 

                                                X_train = np.load(f'data/qubits={n_qubits}/classes={num_classes}/samples={num_samples}/exp={exp}/gs.npy')
                                                y_train = np.load(f'data/qubits={n_qubits}/classes={num_classes}/samples={num_samples}/exp={exp}/label_r={r}.npy')
                                                X_test = np.load(f'data/qubits={n_qubits}/classes={num_classes}/samples=1000/exp=1/gs.npy')
                                                y_test = np.load(f'data/qubits={n_qubits}/classes={num_classes}/samples=1000/exp=1/label_r={r}.npy')

                                                model = QPRVariationalClassifier(n_qubits=n_qubits, var_ansatz=var_ansatz, n_layers=num_layers, max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate,  
                                                                                 convergence_interval=convergence_interval, num_classes=num_classes, num_samples=num_samples, r=r, exp=exp)
                                                model.fit(X_train, y_train)
                                                g, mu_marg_10, mu_marg_Q1, mu_marg_Q2, mu_marg_Q3, mu_marg_mean, mu_param, mu_param_eff1, mu_param_eff10, mu_param_eff100 = model.get_results(X_train, y_train, X_test, y_test)

                                                g_list = np.append(g_list, g)
                                                mu_marg_10_list = np.append(mu_marg_10_list, mu_marg_10 / num_samples)
                                                mu_marg_Q1_list = np.append(mu_marg_Q1_list, mu_marg_Q1 / num_samples)
                                                mu_marg_Q2_list = np.append(mu_marg_Q2_list, mu_marg_Q2 / num_samples)
                                                mu_marg_Q3_list = np.append(mu_marg_Q3_list, mu_marg_Q3 / num_samples)
                                                mu_marg_mean_list = np.append(mu_marg_mean_list, mu_marg_mean / num_samples)
                                                mu_param_list = np.append(mu_param_list, mu_param / num_samples)
                                                mu_param_eff1_list = np.append(mu_param_eff1_list, mu_param_eff1 / num_samples)
                                                mu_param_eff10_list = np.append(mu_param_eff10_list, mu_param_eff10 / num_samples)
                                                mu_param_eff100_list = np.append(mu_param_eff100_list, mu_param_eff100 / num_samples)
                                                mu_marg_10_param_list = np.append(mu_marg_10_param_list, mu_marg_10 * mu_param / num_samples)
                                                mu_marg_10_param_eff1_list = np.append(mu_marg_10_param_eff1_list, mu_marg_10 * mu_param_eff1 / num_samples)
                                                mu_marg_10_param_eff10_list = np.append(mu_marg_10_param_eff10_list, mu_marg_10 * mu_param_eff10 / num_samples)
                                                mu_marg_10_param_eff100_list = np.append(mu_marg_10_param_eff100_list, mu_marg_10 * mu_param_eff100 / num_samples)
                                                mu_marg_Q1_param_list = np.append(mu_marg_Q1_param_list, mu_marg_Q1 * mu_param / num_samples)
                                                mu_marg_Q1_param_eff1_list = np.append(mu_marg_Q1_param_eff1_list, mu_marg_Q1 * mu_param_eff1 / num_samples)
                                                mu_marg_Q1_param_eff10_list = np.append(mu_marg_Q1_param_eff10_list, mu_marg_Q1 * mu_param_eff10 / num_samples)
                                                mu_marg_Q1_param_eff100_list = np.append(mu_marg_Q1_param_eff100_list, mu_marg_Q1 * mu_param_eff100 / num_samples)
                                                mu_marg_Q2_param_list = np.append(mu_marg_Q2_param_list, mu_marg_Q2 * mu_param / num_samples)
                                                mu_marg_Q2_param_eff1_list = np.append(mu_marg_Q2_param_eff1_list, mu_marg_Q2 * mu_param_eff1 / num_samples)
                                                mu_marg_Q2_param_eff10_list = np.append(mu_marg_Q2_param_eff10_list, mu_marg_Q2 * mu_param_eff10 / num_samples)
                                                mu_marg_Q2_param_eff100_list = np.append(mu_marg_Q2_param_eff100_list, mu_marg_Q2 * mu_param_eff100 / num_samples)
                                                mu_marg_Q3_param_list = np.append(mu_marg_Q3_param_list, mu_marg_Q3 * mu_param / num_samples)
                                                mu_marg_Q3_param_eff1_list = np.append(mu_marg_Q3_param_eff1_list, mu_marg_Q3 * mu_param_eff1 / num_samples)
                                                mu_marg_Q3_param_eff10_list = np.append(mu_marg_Q3_param_eff10_list, mu_marg_Q3 * mu_param_eff10 / num_samples)
                                                mu_marg_Q3_param_eff100_list = np.append(mu_marg_Q3_param_eff100_list, mu_marg_Q3 * mu_param_eff100 / num_samples)
                                                mu_marg_mean_param_list = np.append(mu_marg_mean_param_list, mu_marg_mean * mu_param / num_samples)
                                                mu_marg_mean_param_eff1_list = np.append(mu_marg_mean_param_eff1_list, mu_marg_mean * mu_param_eff1 / num_samples)
                                                mu_marg_mean_param_eff10_list = np.append(mu_marg_mean_param_eff10_list, mu_marg_mean * mu_param_eff10 / num_samples)
                                                mu_marg_mean_param_eff100_list = np.append(mu_marg_mean_param_eff100_list, mu_marg_mean * mu_param_eff100 / num_samples)
                                            
                                            
                    N = len(num_layers_list) * len(max_steps_list) * len(batch_size_list) * len(learning_rate_list) * len(convergence_interval_list) * len(num_samples_list) * len(exp_list) 
                    # Make 29 * N Array of Mus
                    mu_array = np.array([mu_param_list, mu_param_eff1_list, mu_param_eff10_list, mu_param_eff100_list,
                                        mu_marg_mean_list, mu_marg_10_list, mu_marg_Q1_list, mu_marg_Q2_list, mu_marg_Q3_list,
                                        mu_marg_mean_param_list, mu_marg_mean_param_eff1_list, mu_marg_mean_param_eff10_list, mu_marg_mean_param_eff100_list,
                                        mu_marg_10_param_list, mu_marg_10_param_eff1_list, mu_marg_10_param_eff10_list, mu_marg_10_param_eff100_list,
                                        mu_marg_Q1_param_list, mu_marg_Q1_param_eff1_list, mu_marg_Q1_param_eff10_list, mu_marg_Q1_param_eff100_list,
                                        mu_marg_Q2_param_list, mu_marg_Q2_param_eff1_list, mu_marg_Q2_param_eff10_list, mu_marg_Q2_param_eff100_list,
                                        mu_marg_Q3_param_list, mu_marg_Q3_param_eff1_list, mu_marg_Q3_param_eff10_list, mu_marg_Q3_param_eff100_list])
                    #Make 29 * 4 Array of Mutual Information
                    MI_array = np.zeros((len(mu_array), 4))

                    N = len(exp_list) * len(num_samples_list) 
                    num_bins1 = int(np.round(np.sqrt(N)))
                    num_bins2 = int(np.round(np.log2(N) + 1))
                    num_bins3 = int(np.round(2 * np.power(N, 1/3)))
                    for i in range(len(mu_array)):
                        MI_array[i, 0] = mutual_information_hist(mu_array[i], g_list, num_bins1)
                        MI_array[i, 1] = mutual_information_hist(mu_array[i], g_list, num_bins2)
                        MI_array[i, 2] = mutual_information_hist(mu_array[i], g_list, num_bins3)
                        MI_array[i, 3] = mutual_information_kde_jitter(mu_array[i], g_list)

                    np.save(model.PATH1 + f"mu_array.npy", mu_array)
                    np.save(model.PATH1 + f"MI_array.npy", MI_array)

                    f = open(model.PATH1 + "MI_array.txt", "w")
                    mi_name = ["mu_param", "mu_param_eff1", "mu_param_eff10", "mu_param_eff100",
                                "mu_marg_mean", "mu_marg_10", "mu_marg_Q1", "mu_marg_Q2", "mu_marg_Q3",
                                "mu_marg_mean * mu_param", "mu_marg_mean * mu_param_eff1", "mu_marg_mean * mu_param_eff10", "mu_marg_mean * mu_param_eff100",
                                "mu_marg_10 * mu_param", "mu_marg_10 * mu_param_eff1", "mu_marg_10 * mu_param_eff10", "mu_marg_10 * mu_param_eff100",
                                "mu_marg_Q1 * mu_param", "mu_marg_Q1 * mu_param_eff1", "mu_marg_Q1 * mu_param_eff10", "mu_marg_Q1 * mu_param_eff100",
                                "mu_marg_Q2 * mu_param", "mu_marg_Q2 * mu_param_eff1", "mu_marg_Q2 * mu_param_eff10", "mu_marg_Q2 * mu_param_eff100",                                    "mu_marg_Q3 * mu_param", "mu_marg_Q3 * mu_param_eff1", "mu_marg_Q3 * mu_param_eff10", "mu_marg_Q3 * mu_param_eff100"]
                    for i in range(len(mu_array)):
                        f.write(mi_name[i] + ":\n")
                        f.write(f"Estimated Mutual Information (Histogram1): {MI_array[i, 0]}\n")
                        f.write(f"Estimated Mutual Information (Histogram2): {MI_array[i, 1]}\n")
                        f.write(f"Estimated Mutual Information (Histogram3): {MI_array[i, 2]}\n")
                        f.write(f"Estimated Mutual Information (KDE): {MI_array[i, 3]}\n")
                    f.close()
                                            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes_list", nargs="+", type=int, default=[2, 4]) 
    parser.add_argument("--r_list", nargs="+", type=float, default=[0.0, 0.5, 1.0])      
    parser.add_argument("--n_qubits_list", nargs="+", type=int, default=[4]) 
    parser.add_argument("--var_ansatz_list", nargs="+", type=str, default=["QCNN_not_shared", "QCNN_shared", "SEL"])

    parser.add_argument("--num_layers_list", nargs="+", type=int, default=[2, 4, 6]) 
    parser.add_argument("--max_steps_list", nargs="+", type=int, default=[2000]) 
    parser.add_argument("--batch_size_list", nargs="+", type=int, default=[10]) 
    parser.add_argument("--learning_rate_list", nargs="+", type=float, default=[1e-3]) 
    parser.add_argument("--convergence_interval_list", nargs="+", type=int, default=[200]) 
    parser.add_argument("--num_samples_list", nargs="+", type=int, default=[20]) 
    parser.add_argument("--exp_list", nargs="+", type=int, default=range(1, 51))         
    args, _ = parser.parse_known_args()
    run_all(**vars(args))