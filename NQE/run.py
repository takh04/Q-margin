from models.IQP import IQPVariationalClassifier
from models.data_reuploading import DataReuploadingClassifier
from models.NQE import NQEClassifier
from models.TQE import TQE1Classifier, TQE3Classifier
import argparse
import data
import torch
import numpy as np

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
                elif classifier == "DataReuploading":
                    model = DataReuploadingClassifier(exp=exp, data=d, batch_size=batch_size, convergence_interval=convergence_interval, max_steps = max_steps)
                    X_train, X_test, y_train, y_test = data.data_load_and_process(d, n_features=8,  classes=[0, 1])
                elif classifier == "NQE":
                    model = NQEClassifier(exp=exp, data=d, batch_size=batch_size, convergence_interval=convergence_interval, max_steps = max_steps)
                    X_train, X_test, y_train, y_test = data.data_load_and_process(d, n_features=8,  classes=[0, 1])
                elif classifier == "TQE1":
                    model = TQE1Classifier(exp=exp, data=d, batch_size=batch_size, convergence_interval=convergence_interval, max_steps = max_steps)
                    X_train, X_test, y_train, y_test = data.data_load_and_process(d, n_features=8,  classes=[0, 1])
                elif classifier == "TQE3":
                    model = TQE3Classifier(exp=exp, data=d, batch_size=batch_size, convergence_interval=convergence_interval, max_steps = max_steps)
                    X_train, X_test, y_train, y_test = data.data_load_and_process(d, n_features=8,  classes=[0, 1])
                
            model.fit(X_train, y_train)
            model.get_results(X_train, y_train, X_test, y_test)
                    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_list", nargs="+", type=int, default=[1])
    parser.add_argument("--data_list", nargs="+", type=str, default=["mnist", "fashion", "kmnist"])
    parser.add_argument("--classifier_list", nargs="+", type=str, default=["IQP", "NQE", "TQE1" ])   
    args, _ = parser.parse_known_args()
    run_all(**vars(args))