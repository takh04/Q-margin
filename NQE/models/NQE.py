import os
import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from torch import nn
from models.model_utils import *

class CNN_NQEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits=8,
        n_repeats=3,
        n_layers=1,
        max_steps=1000,
        batch_size=10,
        learning_rate=1e-3,
        convergence_interval=200,
        data='mnist',
        exp=1,
    ):
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats
        self.n_layers = n_layers
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_interval = convergence_interval
        self.data = data
        self.exp = exp

    def initialize(self, X, y):
        self.PATH = f"results/{self.__class__.__name__}/{self.data}/{self.exp}exp/"
        os.makedirs(self.PATH, exist_ok=True)
        self.model_state_dict = optimize_nqe(self.n_qubits, self.n_repeats, X, y, 'conv', self.PATH)
        self.model = construct_model(self.n_qubits, self.n_repeats, self.n_layers)

    def x_transform(self,X):
        X = torch.tensor(X, dtype=torch.float32)
        class NQE_transform(torch.nn.Module):
            def __init__(self, n_qubits):
                super().__init__()
                self.n_qubits = n_qubits
                self.nn_layer = get_nn_layer(self.n_qubits, "conv")
                    
            def forward(self, x):
                x = self.nn_layer(x)
                return x.detach().numpy()
                
        model = NQE_transform(self.n_qubits)
        model.load_state_dict(torch.load(self.PATH + "model.pt"))
        X_transformed = model(X)
        return X_transformed

    def fit(self, X, y):
        self.initialize(X, y)
        train(
            self,
            self.x_transform(X),
            y,
            self.convergence_interval,
        )

    def predict(self, X):
        X = self.x_transform(X)
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        predictions = torch.argmax(probabilities, dim=1)
        return predictions.detach().numpy()

    def predict_proba(self, X):
        X = self.x_transform(X)
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        return probabilities.detach().numpy()

    def score(self, X, y):
        X = self.x_transform(X)
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        y = torch.from_numpy(y).to(torch.long)
        predictions = torch.argmax(probabilities, dim=1)
        return (predictions == y).float().mean().item()

    def get_results(self, X_train, y_train, X_test, y_test):
        print("Calculating scores and margins...")
        train_acc, margin_dist, margin_boxplot, margin_mean = score_and_margins(self.model, self.x_transform(X_train), y_train)
        test_acc = self.score(X_test, y_test)
        
        generalization_gap = train_acc - test_acc
        #margin_dist, margin_boxplot, margin_mean = self.get_margins(X_train, y_train)
        print("Calculating trace distance...")
        trace_distance = get_trace_distance(self.n_qubits, self.n_repeats, self.x_transform(X_train), y_train)

        f = open(self.PATH + "results.txt", "w")
        f.write(f"Train Accuracy: {train_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Generalization Gap: {generalization_gap}\n")

        np.save(self.PATH + "margin_dist.npy", margin_dist)
        np.save(self.PATH + "margin_boxplot.npy", margin_boxplot)
        np.save(self.PATH + "margin_mean.npy", margin_mean)
        np.save(self.PATH + "trace_distance.npy", trace_distance)

   






#================================================================================================


class PCA_NQEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits=8,
        n_repeats=3,
        n_layers=1,
        max_steps=1000,
        batch_size=10,
        learning_rate=1e-3,
        convergence_interval=200,
        data='mnist',
        exp=1,
    ):
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats
        self.n_layers = n_layers
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_interval = convergence_interval
        self.data = data
        self.exp = exp

    def initialize(self, X, y):
        self.PATH = f"results/{self.__class__.__name__}/{self.data}/{self.exp}exp/"
        os.makedirs(self.PATH, exist_ok=True)
        self.model_state_dict = optimize_nqe(self.n_qubits, self.n_repeats, X, y, "linear", self.PATH)
        self.model = construct_model(self.n_qubits, self.n_repeats, self.n_layers)

    def x_transform(self,X):
        X = torch.tensor(X, dtype=torch.float32)
        class NQE_transform(torch.nn.Module):
            def __init__(self, n_qubits):
                super().__init__()
                self.n_qubits = n_qubits
                self.nn_layer = get_nn_layer(self.n_qubits, "linear")
                    
            def forward(self, x):
                x = self.nn_layer(x)
                return x.detach().numpy()
                
        model = NQE_transform(self.n_qubits)
        model.load_state_dict(torch.load(self.PATH + "model.pt"))
        X_transformed = model(X)
        return X_transformed
    
    def fit(self, X, y):
        self.initialize(X, y)
        train(
            self,
            self.x_transform(X),
            y,
            self.convergence_interval,
        )

    def predict(self, X):
        X = self.x_transform(X)
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        predictions = torch.argmax(probabilities, dim=1)
        return predictions.detach().numpy()

    def predict_proba(self, X):
        X = self.x_transform(X)
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        return probabilities.detach().numpy()

    def score(self, X, y):
        X = self.x_transform(X)
        with torch.no_grad():
            probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        y = torch.from_numpy(y).to(torch.long)
        predictions = torch.argmax(probabilities, dim=1)
        return (predictions == y).float().mean().item()

    def get_results(self, X_train, y_train, X_test, y_test):
        print("Calculating scores and margins...")
        train_acc, margin_dist, margin_boxplot, margin_mean = score_and_margins(self.model, self.x_transform(X_train), y_train)
        test_acc = self.score(X_test, y_test)
        
        generalization_gap = train_acc - test_acc
        print("Calculating trace distance...")
        trace_distance = get_trace_distance(self.n_qubits, self.n_repeats, self.x_transform(X_train), y_train)

        f = open(self.PATH + "results.txt", "w")
        f.write(f"Train Accuracy: {train_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Generalization Gap: {generalization_gap}\n")

        np.save(self.PATH + "margin_dist.npy", margin_dist)
        np.save(self.PATH + "margin_boxplot.npy", margin_boxplot)
        np.save(self.PATH + "margin_mean.npy", margin_mean)
        np.save(self.PATH + "trace_distance.npy", trace_distance)