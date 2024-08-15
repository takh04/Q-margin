import os
import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from models.model_utils import *


def construct_model(n_qubits, n_layers, var_ansatz, num_classes):
    dev = qml.device("default.qubit", wires=n_qubits)

    if num_classes == 2:
        meas_wires = [0]
    elif num_classes == 4:
        meas_wires = [0,2]
    

    if var_ansatz == "SEL": 
        @qml.qnode(dev)
        def circuit(inputs, params):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(
                params, wires=range(n_qubits), imprimitive=qml.CZ
            )
            return qml.probs(wires=meas_wires)
        num_params = n_layers * n_qubits * 3
        weight_shapes = {"params": (n_layers, n_qubits, 3)}


    elif var_ansatz == "QCNN_shared":
        @qml.qnode(dev)
        def circuit(inputs, params):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits))
            get_qcnn(params, n_qubits, n_layers, parameter_sharing=True)
            return qml.probs(wires=meas_wires)
        
        num_params = get_num_params(n_qubits, n_layers, True)
        weight_shapes = {"params": (num_params)}


    elif var_ansatz == "QCNN_not_shared":
        @qml.qnode(dev)
        def circuit(inputs, params):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits))
            get_qcnn(params, n_qubits, n_layers, parameter_sharing=False)
            return qml.probs(wires=meas_wires)
        
        num_params = get_num_params(n_qubits, n_layers, False)
        weight_shapes = {"params": (num_params)}
    
    
    class model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        def forward(self, x):
            return self.qlayer(x)
    model = model()
    return num_params, model


class QPRVariationalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits=4,
        var_ansatz="SEL",
        n_layers=10,
        max_steps=400,
        batch_size=32,
        learning_rate=0.001,
        convergence_interval=20,
        
        num_classes=4,
        num_samples=100,
        r = 0.0,
        exp = 1,
    ):
        self.n_qubits_ = n_qubits
        self.var_ansatz = var_ansatz
        self.n_layers = n_layers
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_interval = convergence_interval

        self.n_classes_ = num_classes
        self.num_samples = num_samples
        self.r = r
        self.exp = exp


    def initialize(self):
        self.PATH1 = f"results/{self.n_classes_}C/{self.r}R/{self.n_qubits_}Q/{self.var_ansatz}/"
        self.PATH2 = f"{self.n_layers}L_{self.max_steps}MS_{self.batch_size}BS_{self.learning_rate}LR_{self.convergence_interval}conv/{self.num_samples}S/{self.exp}E/"
        os.makedirs(self.PATH1 + self.PATH2, exist_ok=True)
        self.num_params, self.model = construct_model(self.n_qubits_, self.n_layers, self.var_ansatz, self.n_classes_)
        for param in self.model.parameters():
            self.weight_init = param.detach().numpy()

    def fit(self, X, y):
        self.initialize()
        train(
            self,
            X,
            y,
            self.convergence_interval,
        )
    
    def predict(self, X):
        probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        predictions = torch.argmax(probabilities, dim=1)
        return predictions.detach().numpy()

    def predict_proba(self, X):
        probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        return probabilities.detach().numpy()
    
    def score(self, X, y):
        probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        y = torch.from_numpy(y).to(torch.long)
        predictions = torch.argmax(probabilities, dim=1)
        return (predictions == y).float().mean().item()
    
    def get_margins(self, X, y):
        probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        y = torch.from_numpy(y).to(torch.long)

        top_two_probs, top_two_indices = torch.topk(probabilities, 2, dim=1)
        R = top_two_probs[:, 0] - top_two_probs[:, 1]
        predicted_labels = torch.argmax(probabilities, dim=1)

        margin_dist = torch.where(predicted_labels == y, R, torch.zeros_like(R))
        margin_mean = margin_dist.mean().item()
        margin_Q1 = torch.quantile(margin_dist, 0.25).item()
        margin_Q2 = torch.quantile(margin_dist, 0.50).item()
        margin_Q3 = torch.quantile(margin_dist, 0.75).item()
        margin_IQR = margin_Q3 - margin_Q1
        margin_boxplot = np.array([margin_Q1 - 1.5 * margin_IQR, margin_Q1, margin_Q2, margin_Q3, margin_Q3 + 1.5 * margin_IQR])
        return margin_dist.detach().numpy(), margin_boxplot, margin_Q1, margin_Q2, margin_Q3, margin_mean
    
    def get_mu_params(self, X, y):
        mu_param = self.num_params * np.log(self.num_params)

        def count_eff(init, final, threshold):
            changes = np.abs(final - init)
            num_changes = np.sum(changes > threshold)
            return num_changes
        
        mu_param_eff1 = count_eff(self.weight_init, self.weight_final, 0.5)
        mu_param_eff2 = count_eff(self.weight_init, self.weight_final, 0.1)
        return mu_param, mu_param_eff1, mu_param_eff2
    
    def get_results(self, X_train, y_train, X_test, y_test):
        train_acc = self.score(X_train, y_train)
        test_acc = self.score(X_test, y_test)
        generalization_gap = train_acc - test_acc
        margin_dist, margin_boxplot, mu_marg_Q1, mu_marg_Q2, mu_marg_Q3, mu_marg_mean = self.get_margins(X_train, y_train)

        mu_param, mu_param_eff1, mu_param_eff2 = self.get_mu_params(X_train, y_train)

        f = open(self.PATH1 + self.PATH2 + "results.txt", "w")
        f.write(f"Train Accuracy: {train_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Generalization Gap: {generalization_gap}\n")
        f.write(f"Margin Q1: {mu_marg_Q1}\n")
        f.write(f"Margin Q2: {mu_marg_Q2}\n")
        f.write(f"Margin Q3: {mu_marg_Q3}\n")
        f.write(f"Margin Mean: {mu_marg_mean}\n")
        f.write(f"Mu Params: {mu_param}\n")
        f.write(f"Mu Params Eff1: {mu_param_eff1}\n")
        f.write(f"Mu Params Eff2: {mu_param_eff2}\n")
        f.close()
        
        np.save(self.PATH1 + self.PATH2 + "margin_dist.npy", margin_dist)
        np.save(self.PATH1 + self.PATH2 + "margin_boxplot.npy", margin_boxplot)
        np.save(self.PATH1 + self.PATH2 + "mu_marg_Q1.npy", mu_marg_Q1)
        np.save(self.PATH1 + self.PATH2 + "mu_marg_Q2.npy", mu_marg_Q2)
        np.save(self.PATH1 + self.PATH2 + "mu_marg_Q3.npy", mu_marg_Q3)
        np.save(self.PATH1 + self.PATH2 + "mu_marg_mean.npy", mu_marg_mean)
        np.save(self.PATH1 + self.PATH2 + "mu_param.npy", mu_param)
        np.save(self.PATH1 + self.PATH2 + "mu_param_eff1.npy", mu_param_eff1)
        np.save(self.PATH1 + self.PATH2 + "mu_param_eff2.npy", mu_param_eff2)
        np.save(self.PATH1 + self.PATH2 + "train_acc.npy", np.array(train_acc))
        np.save(self.PATH1 + self.PATH2 + "test_acc.npy", np.array(test_acc))
        np.save(self.PATH1 + self.PATH2 + "generalization_gap.npy", np.array(generalization_gap))
        
        return generalization_gap, train_acc, test_acc, mu_marg_Q1, mu_marg_Q2, mu_marg_Q3, mu_marg_mean, mu_param, mu_param_eff1, mu_param_eff2