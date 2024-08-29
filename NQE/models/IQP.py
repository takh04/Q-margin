import os
import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from models.model_utils import *

def construct_model(n_qubits, n_repeats, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)
    meas_wires = [0]
    
    @qml.qnode(dev)
    def circuit(inputs, params):
        qml.IQPEmbedding(inputs, n_repeats=n_repeats, wires=range(n_qubits))
        get_qcnn(params, n_qubits, n_layers, parameter_sharing=True)
        return qml.probs(wires=meas_wires)
        
    num_params = get_num_params(n_qubits, n_layers, True)
    weight_shapes = {"params": (num_params)}
    
    
    class model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        def forward(self, x):
            return self.qlayer(x)
    model = model()
    return model


class IQPVariationalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_repeats=3,
        n_layers=1,
        max_steps=1000,
        batch_size=10,
        learning_rate=1e-3,
        convergence_interval=200,
        data = 'mnist',
        exp = 1,
    ):
        self.n_repeats = n_repeats
        self.n_layers = n_layers
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
        self.model = construct_model(self.n_qubits_, self.n_repeats, self.n_layers)

    def fit(self, X, y):
        self.initialize(X, y)
        train(
            self,
            X,
            y,
            self.convergence_interval,
        )
    
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

    def score_and_margins(self, X, y, batch_size=1024):
        num_samples = X.shape[0]
        correct_predictions = 0
        total_samples = 0
        margin_dists = []

        for i in range(0, num_samples, batch_size):
            # Extract the batch
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Convert to tensors
            X_tensor = torch.from_numpy(X_batch).to(torch.float32)
            y_tensor = torch.from_numpy(y_batch).to(torch.long)

            # Get probabilities for the batch
            with torch.no_grad():
                probabilities = self.model(X_tensor)
            
            # Score calculation
            predictions = torch.argmax(probabilities, dim=1)
            correct_predictions += (predictions == y_tensor).sum().item()
            total_samples += y_batch.shape[0]

            # Margin calculation
            correct_label_probs = probabilities.gather(1, y_tensor.view(-1, 1)).squeeze(1)
            incorrect_label_probs, _ = torch.max(probabilities.masked_fill(torch.eye(probabilities.size(1))[y_tensor].bool(), float('-inf')), dim=1)
            margin_dist_batch = correct_label_probs - incorrect_label_probs

            # Zero out margins for incorrect predictions
            margin_dist_batch = torch.where(predictions == y_tensor, margin_dist_batch, torch.zeros_like(margin_dist_batch))
            margin_dists.append(margin_dist_batch)

        # Concatenate all margins
        margin_dist = torch.cat(margin_dists, dim=0)

        # Calculate overall accuracy
        accuracy = correct_predictions / total_samples

        # Calculate margin statistics
        margin_mean = margin_dist.mean().item()
        margin_min = margin_dist.min().item()
        margin_Q1 = torch.quantile(margin_dist, 0.25).item()
        margin_Q2 = torch.quantile(margin_dist, 0.50).item()
        margin_Q3 = torch.quantile(margin_dist, 0.75).item()
        margin_max = margin_dist.max().item()
        margin_boxplot = np.array([margin_min, margin_Q1, margin_Q2, margin_Q3, margin_max])

        # Return both accuracy and margin-related outputs
        return accuracy, margin_dist.detach().numpy(), margin_boxplot, margin_mean


    
    def get_results(self, X_train, y_train, X_test, y_test):
        print("Calculating scores and margins...")
        train_acc, margin_dist, margin_boxplot, margin_mean = self.score_and_margins(X_train, y_train)
        print("Calculating test accuracy...")
        test_acc = self.score(X_test, y_test)
        
        generalization_gap = train_acc - test_acc
        #margin_dist, margin_boxplot, margin_mean = self.get_margins(X_train, y_train)
        print("Calculating trace distance...")
        trace_distance = get_trace_distance(self.n_qubits_, self.n_repeats, self.n_layers,  X_train, y_train, 1024)

        f = open(self.PATH + "results.txt", "w")
        f.write(f"Train Accuracy: {train_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Generalization Gap: {generalization_gap}\n")
        
        np.save(self.PATH + "margin_dist.npy", margin_dist)
        np.save(self.PATH + "margin_boxplot.npy", margin_boxplot)
        np.save(self.PATH + "margin_mean.npy", margin_mean)
        np.save(self.PATH + "trace_distance.npy", trace_distance)
        
        