from models.IQP import IQPVariationalClassifier
from models.NQE import CNN_NQEClassifier, PCA_NQEClassifier
from models.TQE import TQEClassifier
import argparse
import data
import torch
import numpy as np
import os

n_layer = 1
max_steps = 5000
convergence_interval = 300
batch_size = 16
def run_all(exp_list, data_list, classifier_list):
    for exp in exp_list: 
        for d in data_list:
            for classifier in classifier_list:
                if classifier == "IQP":
                    model = IQPVariationalClassifier(exp=exp, data=d, batch_size=batch_size, convergence_interval=convergence_interval, max_steps = max_steps)
                    X_train, X_test, y_train, y_test = data.data_load_and_process(d, n_features=8,  classes=[0, 1])
                elif classifier == "TQE":
                    model = TQEClassifier(exp=exp, data=d, batch_size=batch_size, convergence_interval=convergence_interval, max_steps = max_steps)
                    X_train, X_test, y_train, y_test = data.data_load_and_process(d, n_features=8,  classes=[0, 1])
                elif classifier == "PCA_NQE":
                    model = PCA_NQEClassifier(exp=exp, data=d, batch_size=batch_size, convergence_interval=convergence_interval, max_steps = max_steps)
                    X_train, X_test, y_train, y_test = data.data_load_and_process(d, n_features=8,  classes=[0, 1])
                elif classifier == "CNN_NQE":
                    model = CNN_NQEClassifier(exp=exp, data=d, batch_size=batch_size, convergence_interval=convergence_interval, max_steps = max_steps)
                    X_train, X_test, y_train, y_test = data.data_load_and_process(d, n_features=None,  classes=[0, 1])
                    
                PATH = f"results/{model.__class__.__name__}/{d}/{exp}exp/"
                if os.path.exists(PATH + "results.txt"):
                    print(f"Skipping {PATH}")
                else:
                    model.fit(X_train, y_train)
                    model.get_results(X_train, y_train, X_test, y_test)
                    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_list", nargs="+", type=int, default=[1])
    parser.add_argument("--data_list", nargs="+", type=str, default=["mnist", "fashion", "kmnist"])
    parser.add_argument("--classifier_list", nargs="+", type=str, default=["IQP", "TQE", "PCA_NQE", "CNN_NQE"])   
    args, _ = parser.parse_known_args()
    run_all(**vars(args))