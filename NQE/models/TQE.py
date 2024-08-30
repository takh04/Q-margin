import os
import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from models.model_utils import *

class TQEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_repeats=3,
        n_layers=1,
        n_layers_TQE=1,
        max_steps=1000,
        batch_size=10,
        learning_rate=1e-3,
        convergence_interval=200,
        data = 'mnist',
        exp = 1,
    ):
        self.n_repeats = n_repeats
        self.n_layers = n_layers
        self.n_layers_TQE = n_layers_TQE
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_interval = convergence_interval
        self.data = data
        self.exp = exp


    def initialize(self, X, y):
        self.n_qubits_ = X.shape[1]
        self.PATH= f"results/{self.__class__.__name__}/{self.data}/{self.exp}exp/"
        os.makedirs(self.PATH, exist_ok=True)
        self.model = construct_model_TQE(self.n_qubits_, self.n_repeats, self.n_layers, self.n_layers_TQE)

    def fit(self, X, y):
        self.initialize(X, y)
        train(
            self,
            X,
            y,
            self.convergence_interval,
        )
        parameters_list = [param.detach().numpy() for param in self.model.parameters()]
        self.trained_params_embedding = np.concatenate(parameters_list[:self.n_layers_TQE * 2 * self.n_qubits_])
    
    def predict(self, X):
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        predictions = torch.argmax(probabilities, dim=1)
        return predictions.detach().numpy()

    def predict_proba(self, X):
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        return probabilities.detach().numpy()
    
    def score(self, X, y):
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        y = torch.from_numpy(y).to(torch.long)
        predictions = torch.argmax(probabilities, dim=1)
        return (predictions == y).float().mean().item()

    def get_results(self, X_train, y_train, X_test, y_test):
        print("Calculating scores and margins...")
        train_acc, margin_dist, margin_boxplot, margin_mean = score_and_margins(self.model, X_train, y_train)
        test_acc = self.score(X_test, y_test)
        
        generalization_gap = train_acc - test_acc
        #margin_dist, margin_boxplot, margin_mean = self.get_margins(X_train, y_train)
        print("Calculating trace distance...")
        trace_distance = get_trace_distance_TQE(self.n_qubits_, self.n_repeats,  X_train, y_train, self.n_layers_TQE, self.trained_params_embedding,)

        f = open(self.PATH + "results.txt", "w")
        f.write(f"Train Accuracy: {train_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Generalization Gap: {generalization_gap}\n")
        
        np.save(self.PATH + "margin_dist.npy", margin_dist)
        np.save(self.PATH + "margin_boxplot.npy", margin_boxplot)
        np.save(self.PATH + "margin_mean.npy", margin_mean)
        np.save(self.PATH + "trace_distance.npy", trace_distance)