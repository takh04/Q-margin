import os
import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from torch import nn
from models.model_utils import *

def get_trace_distance(n_qubits, n_repeats, n_layers, X_train, Y_train):
    x1 = torch.tensor(X_train[Y_train == 1], dtype=torch.float32)
    x0 = torch.tensor(X_train[Y_train == 0], dtype=torch.float32)

    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(inputs):
        qml.IQPEmbedding(inputs, n_repeats=n_repeats, wires=range(n_qubits))
        return qml.density_matrix(wires=range(n_qubits))
    
    class Distance(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes={})
        def forward(self, x0, x1):
            rhos1 = self.qlayer(x1)
            rhos0 = self.qlayer(x0)

            rho1 = torch.sum(rhos1, dim=0) / len(x1)
            rho0 = torch.sum(rhos0, dim=0) / len(x0)
            rho_diff = rho1 - rho0
            eigvals = torch.linalg.eigvals(rho_diff)
            return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))

    model = Distance()
    return model(x0, x1).item()

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


class NQEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_repeats=3,
        n_layers=1,
        max_steps=1000,
        batch_size=10,
        learning_rate=1e-3,
        convergence_interval=200,
        data='mnist',
        exp=1,
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
        self.n_qubits = X.shape[1]
        self.PATH = f"results/{self.__class__.__name__}/{self.data}/{self.exp}exp/"
        os.makedirs(self.PATH, exist_ok=True)
        self.model_state_dict = self.optimize_nqe(X, y)
        self.model = construct_model(self.n_qubits, self.n_repeats, self.n_layers)

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
        probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        predictions = torch.argmax(probabilities, dim=1)
        return predictions.detach().numpy()

    def predict_proba(self, X):
        X = self.x_transform(X)
        probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        return probabilities.detach().numpy()

    def score(self, X, y):
        X = self.x_transform(X)
        probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        y = torch.from_numpy(y).to(torch.long)
        predictions = torch.argmax(probabilities, dim=1)
        return (predictions == y).float().mean().item()

    def get_margins(self, X, y):
        X = self.x_transform(X)
        probabilities = self.model(torch.from_numpy(X).to(torch.float32))
        y = torch.from_numpy(y).to(torch.long)

        top_two_probs, top_two_indices = torch.topk(probabilities, 2, dim=1)
        R = top_two_probs[:, 0] - top_two_probs[:, 1]
        predicted_labels = torch.argmax(probabilities, dim=1)

        margin_dist = torch.where(predicted_labels == y, R, torch.zeros_like(R))
        margin_mean = margin_dist.mean().item()
        margin_min = margin_dist.min().item()
        margin_Q1 = torch.quantile(margin_dist, 0.25).item()
        margin_Q2 = torch.quantile(margin_dist, 0.50).item()
        margin_Q3 = torch.quantile(margin_dist, 0.75).item()
        margin_max = margin_dist.max().item()
        #margin_IQR = margin_Q3 - margin_Q1
        margin_boxplot = np.array([margin_min, margin_Q1, margin_Q2, margin_Q3, margin_max])
        return margin_dist.detach().numpy(), margin_boxplot, margin_mean

    def get_results(self, X_train, y_train, X_test, y_test):
        train_acc = self.score(X_train, y_train)
        test_acc = self.score(X_test, y_test)

        generalization_gap = train_acc - test_acc
        margin_dist, margin_boxplot, margin_mean = self.get_margins(X_train, y_train)
        X_train_transformed = self.x_transform(X_train)
        trace_distance = get_trace_distance(self.n_qubits, self.n_repeats, self.n_layers, X_train_transformed, y_train)

        f = open(self.PATH + "results.txt", "w")
        f.write(f"Train Accuracy: {train_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Generalization Gap: {generalization_gap}\n")

        np.save(self.PATH + "margin_dist.npy", margin_dist)
        np.save(self.PATH + "margin_boxplot.npy", margin_boxplot)
        np.save(self.PATH + "margin_mean.npy", margin_mean)
        np.save(self.PATH + "trace_distance.npy", trace_distance)

    def optimize_nqe(self, X, y):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        def new_data(batch_size, X, Y):
            X1_new, X2_new, Y_new = [], [], []
            for i in range(batch_size):
                n, m = np.random.randint(len(X)), np.random.randint(len(X))
                X1_new.append(X[n])
                X2_new.append(X[m])
                if Y[n] == Y[m]:
                    Y_new.append(1)
                else:
                    Y_new.append(0)
            X1_new, X2_new, Y_new = np.array(X1_new), np.array(X2_new), np.array(Y_new)
            return torch.tensor(X1_new, dtype=torch.float32), torch.tensor(X2_new, dtype=torch.float32), torch.tensor(Y_new)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs): 
            qml.IQPEmbedding(inputs[0:self.n_qubits], n_repeats=self.n_repeats, wires=range(self.n_qubits))
            qml.adjoint(qml.IQPEmbedding)(inputs[self.n_qubits:2*self.n_qubits], n_repeats=self.n_repeats, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        class NQE_optimize(torch.nn.Module):
            def __init__(self, n_qubits):
                super().__init__()
                self.qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
                self.linear_relu_stack1 = nn.Sequential(
                    nn.Linear(n_qubits, 2 * n_qubits),
                    nn.ReLU(),
                    nn.Linear(2 * n_qubits, 4 * n_qubits),
                    nn.ReLU(),
                    nn.Linear(4 * n_qubits, 4 * n_qubits),
                    nn.ReLU(),
                    nn.Linear(4 * n_qubits, 2 * n_qubits),
                    nn.ReLU(),
                    nn.Linear(2 * n_qubits, n_qubits),
                )
            def forward(self, x1, x2):
                x1 = self.linear_relu_stack1(x1)
                x2 = self.linear_relu_stack1(x2)
                x = torch.concat([x1, x2], 1)
                x = self.qlayer1(x)
                return x[:,0]
            
        def train_models():
            model = NQE_optimize(self.n_qubits)
            model.train()
            loss_fn = torch.nn.MSELoss()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            loss_history = []
            for it in range(self.max_steps):
                X1_batch, X2_batch, Y_batch = new_data(self.batch_size, X, y)
                pred = model(X1_batch, X2_batch)
                pred, Y_batch = pred.to(torch.float32), Y_batch.to(torch.float32)
                loss = loss_fn(pred, Y_batch)
                loss_history.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                if it % 1000 == 0:
                    print(f"Iterations: {it} Loss: {loss.item()}")

                if it > 2 * self.convergence_interval:
                    average1 = np.mean(loss_history[-self.convergence_interval:])
                    average2 = np.mean(loss_history[-2 * self.convergence_interval:-self.convergence_interval])
                    std1 = np.std(loss_history[-self.convergence_interval:])
                    if np.abs(average1 - average2) < std1 / np.sqrt(self.convergence_interval) / 2:
                        print(f"Convergence reached at step {it}")
                        break
                    
            print("NQE Optimization Complete")
            torch.save(model.state_dict(), self.PATH + "model.pt")
            return model.state_dict()
        
        train_models()

    def x_transform(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        class NQE_transform(torch.nn.Module):
            def __init__(self, n_qubits):
                super().__init__()
                self.n_qubits = n_qubits
                self.linear_relu_stack1 = nn.Sequential(
                    nn.Linear(n_qubits, 2 * n_qubits),
                    nn.ReLU(),
                    nn.Linear(2 * n_qubits, 4 * n_qubits),
                    nn.ReLU(),
                    nn.Linear(4 * n_qubits, 4 * n_qubits),
                    nn.ReLU(),
                    nn.Linear(4 * n_qubits, 2 * n_qubits),
                    nn.ReLU(),
                    nn.Linear(2 * n_qubits, n_qubits)
                )
            def forward(self, x):
                x = self.linear_relu_stack1(x)
                return x.detach().numpy()
            
        model = NQE_transform(self.n_qubits)
        model.load_state_dict(torch.load(self.PATH + "model.pt"))
        X_transformed = model(X)
        return X_transformed

        
        