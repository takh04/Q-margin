import logging
import time
import numpy as np
import torch
import pennylane as qml

# Model Training Utility
def train(self, X, y, convergence_interval=100):
    self.model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    loss_history = []
    start = time.time()
    for step in range(self.max_steps):
          
        batch_index = np.random.randint(0, len(X), (self.batch_size,))
        X_batch = torch.tensor(X[batch_index], dtype=torch.float32)
        y_batch = torch.tensor(y[batch_index], dtype=torch.long)
        
        
        pred = self.model(X_batch)
        loss = loss_fn(pred, y_batch)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        logging.debug(f"{step} - loss: {loss.item()}")

        if np.isnan(loss.item()):
            logging.info(f"nan encountered. Training aborted.")
            break
        
        if step % 1000 == 0:
            print(f"Step {step}, Loss {loss.item()}")
        
        if step > 2 * convergence_interval:

            average1 = np.mean(loss_history[-convergence_interval:])
            average2 = np.mean(loss_history[-2 * convergence_interval:-convergence_interval])
            std1 = np.std(loss_history[-convergence_interval:])
            if np.abs(average1 - average2) < std1 / np.sqrt(convergence_interval) / 2:
                print(f"Convergence reached at step {step}")
                break
            
    end = time.time()
    print(f"Training took {end - start} seconds.")
    loss_history = np.array(loss_history)
    np.save(self.PATH + "loss_history", loss_history)
    torch.save(self.model.state_dict(), self.PATH + "model_params.pth")

#====================================================================================================

# Construct Model Utility

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

def construct_model_TQE(n_qubits, n_repeats, n_layers, n_layers_TQE):
    dev = qml.device("default.qubit", wires=n_qubits)
    meas_wires = [0]
    num_params = get_num_params(n_qubits, n_layers, False)

    def TQE_ansatz(params):
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(n_qubits - 1):
            qml.IsingYY(params[i+n_qubits], wires=[i, i + 1])
        qml.IsingYY(params[2 * n_qubits - 1], wires=[n_qubits - 1, 0])
    
    @qml.qnode(dev)
    def circuit(inputs, params):
        for i in range(n_layers_TQE):   
            TQE_ansatz(params[i * 2 * n_qubits: (i + 1) * 2 * n_qubits])
            qml.IQPEmbedding(inputs, n_repeats=n_repeats, wires=range(n_qubits))
        get_qcnn(params[(n_layers_TQE) * 2 * n_qubits: (n_layers_TQE) * 2 * n_qubits + num_params], n_qubits, n_layers, parameter_sharing=False)
        return qml.probs(wires=meas_wires)
        
    
    weight_shapes = {"params": (n_layers_TQE * 2 * n_qubits + num_params)}
    
    class model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        def forward(self, x):
            return self.qlayer(x)
    model = model()
    return model


#====================================================================================================

# QCNN Utility
def U_SU4(params, wires): # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


def QCNN4_shared(params, num_layers):
# 15 * (num_layers + 1) params
    for i in range(num_layers):
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[0,1])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[2,3])

        U_SU4(params[15 * i : 15 * (i + 1)], wires=[3,0])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[1,2])
    
    U_SU4(params[15 * num_layers : 15 * (num_layers + 1)], wires=[2,0])

def QCNN4_not_shared(params, num_layers):
# 60 * num_layers + 15 params
    for i in range(num_layers):
        U_SU4(params[60 * i : 60 * i + 15], wires=[0,1])
        U_SU4(params[60 * i + 15 : 60 * i + 30], wires=[2,3])
        U_SU4(params[60 * i + 30 : 60 * i + 45], wires=[3,0])
        U_SU4(params[60 * i + 45 : 60 * i + 60], wires=[1,2])
    
    U_SU4(params[60 * num_layers : 60 * num_layers + 15], wires=[2,0])


def QCNN8_shared(params, num_layers):
# 15 * (2 * num_layers + 1) params
    for i in range(num_layers):
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[0,1])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[2,3])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[4,5])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[6,7])

        U_SU4(params[15 * i : 15 * (i + 1)], wires=[7,0])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[1,2])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[3,4])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[5,6])

    for i in range(num_layers):
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[0,2])
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[4,6])

        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[6,0])
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[2,4])

    U_SU4(params[15 * 2 * num_layers : 15 * (2 * num_layers + 1)], wires=[4,0])

def QCNN8_not_shared(params, num_layers):
# 180 * num_layers + 15 params
    for i in range(num_layers):
        U_SU4(params[120 * i : 120 * i + 15], wires=[0,1])
        U_SU4(params[120 * i + 15 : 120 * i + 30], wires=[2,3])
        U_SU4(params[120 * i + 30 : 120 * i + 45], wires=[4,5])
        U_SU4(params[120 * i + 45 : 120 * i + 60], wires=[6,7])

        U_SU4(params[120 * i + 60 : 120 * i + 75], wires=[7,0])
        U_SU4(params[120 * i + 75 : 120 * i + 90], wires=[1,2])
        U_SU4(params[120 * i + 90 : 120 * i + 105], wires=[3,4])
        U_SU4(params[120 * i + 105 : 120 * i + 120], wires=[5,6])

    for i in range(num_layers):
        U_SU4(params[120 * num_layers + 60 * i : 120 * num_layers + 60 * i + 15], wires=[0,2])
        U_SU4(params[120 * num_layers + 60 * i + 15 : 120 * num_layers + 60 * i + 30], wires=[4,6])

        U_SU4(params[120 * num_layers + 60 * i + 30 : 120 * num_layers + 60 * i + 45], wires=[6,0])
        U_SU4(params[120 * num_layers + 60 * i + 45 : 120 * num_layers + 60 * i + 60], wires=[2,4])

    U_SU4(params[120 * num_layers + 60 * num_layers : 120 * num_layers + 60 * num_layers + 15], wires=[4,0])


def QCNN10_shared(params, num_layers):
# 15 * (2 * num_layers + 2) params
    for i in range(num_layers):
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[0,1])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[2,3])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[4,5])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[6,7])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[8,9])

        U_SU4(params[15 * i : 15 * (i + 1)], wires=[9,0])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[1,2])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[3,4])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[5,6])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[7,8])

    for i in range(num_layers):
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[0,2])
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[4,6])
        
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[8,0])
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[2,4])

    U_SU4(params[15 * 2 * num_layers : 15 * (2 * num_layers + 1)], wires=[4,6])

    U_SU4(params[15 * (2 * num_layers + 1) : 15 * (2 * num_layers + 2)], wires=[6,0])


def QCNN10_not_shared(params, num_layers):
# 210 * num_layers + 30
    for i in range(num_layers):
        U_SU4(params[150 * i : 150 * i + 15], wires=[0,1])
        U_SU4(params[150 * i + 15 : 150 * i + 30], wires=[2,3])
        U_SU4(params[150 * i + 30 : 150 * i + 45], wires=[4,5])
        U_SU4(params[150 * i + 45 : 150 * i + 60], wires=[6,7])
        U_SU4(params[150 * i + 60 : 150 * i + 75], wires=[8,9])

        U_SU4(params[150 * i + 75 : 150 * i + 90], wires=[9,0])
        U_SU4(params[150 * i + 90 : 150 * i + 105], wires=[1,2])
        U_SU4(params[150 * i + 105 : 150 * i + 120], wires=[3,4])
        U_SU4(params[150 * i + 120 : 150 * i + 135], wires=[5,6])
        U_SU4(params[150 * i + 135 : 150 * i + 150], wires=[7,8])

    for i in range(num_layers):
        U_SU4(params[150 * num_layers + 60 * i : 150 * num_layers + 60 * i + 15], wires=[0,2])
        U_SU4(params[150 * num_layers + 60 * i + 15 : 150 * num_layers + 60 * i + 30], wires=[4,6])
        
        U_SU4(params[150 * num_layers + 60 * i + 30 : 150 * num_layers + 60 * i + 45], wires=[8,0])
        U_SU4(params[150 * num_layers + 60 * i + 45 : 150 * num_layers + 60 * i + 60], wires=[2,4])
        

    U_SU4(params[210 * num_layers : 210 * num_layers + 15], wires=[4,6])

    U_SU4(params[210 * num_layers + 15 : 210 * num_layers + 30], wires=[6,0])


def QCNN12_shared(params, num_layers):
# 15 * (2 * num_layers + 2) params
    for i in range(num_layers):
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[0,1])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[2,3])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[4,5])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[6,7])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[8,9])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[10,11])

        U_SU4(params[15 * i : 15 * (i + 1)], wires=[11,0])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[1,2])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[3,4])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[5,6])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[7,8])
        U_SU4(params[15 * i : 15 * (i + 1)], wires=[9,10])

    for i in range(num_layers):
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[0,2])
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[4,6])
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[8,10])
        
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[10,0])
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[2,4])
        U_SU4(params[15 * (num_layers + i) : 15 * (num_layers + i + 1)], wires=[6,8])

    U_SU4(params[15 * 2 * num_layers : 15 * (2 * num_layers + 1)], wires=[4,8])

    U_SU4(params[15 * (2 * num_layers + 1) : 15 * (2 * num_layers + 2)], wires=[8,0])


def QCNN12_not_shared(params, num_layers):
# 270 * num_layers + 30
    for i in range(num_layers):
        U_SU4(params[150 * i : 150 * i + 15], wires=[0,1])
        U_SU4(params[150 * i + 15 : 150 * i + 30], wires=[2,3])
        U_SU4(params[150 * i + 30 : 150 * i + 45], wires=[4,5])
        U_SU4(params[150 * i + 45 : 150 * i + 60], wires=[6,7])
        U_SU4(params[150 * i + 60 : 150 * i + 75], wires=[8,9])
        U_SU4(params[150 * i + 75 : 150 * i + 90], wires=[10,11])

        U_SU4(params[150 * i + 90 : 150 * i + 105], wires=[11,0])
        U_SU4(params[150 * i + 105 : 150 * i + 120], wires=[1,2])
        U_SU4(params[150 * i + 120 : 150 * i + 135], wires=[3,4])
        U_SU4(params[150 * i + 135 : 150 * i + 150], wires=[5,6])
        U_SU4(params[150 * i + 150 : 150 * i + 165], wires=[7,8])
        U_SU4(params[150 * i + 165 : 150 * i + 180], wires=[9,10])

    for i in range(num_layers):
        U_SU4(params[180 * num_layers + 90 * i : 180 * num_layers + 90 * i + 15], wires=[0,2])
        U_SU4(params[180 * num_layers + 90 * i + 15 : 180 * num_layers + 90 * i + 30], wires=[4,6])
        U_SU4(params[180 * num_layers + 90 * i + 30 : 180 * num_layers + 90 * i + 45], wires=[8,10])
        
        U_SU4(params[180 * num_layers + 90 * i + 45 : 150 * num_layers + 90 * i + 60], wires=[10,0])
        U_SU4(params[180 * num_layers + 90 * i + 60 : 150 * num_layers + 90 * i + 75], wires=[2,4])
        U_SU4(params[180 * num_layers + 90 * i + 75 : 150 * num_layers + 90 * i + 90], wires=[6,8])
        

    U_SU4(params[270 * num_layers : 270 * num_layers + 15], wires=[4,8])

    U_SU4(params[270 * num_layers + 15 : 270 * num_layers + 30], wires=[8,0])

def get_num_params(num_wires, num_layers, parameter_sharing):
    if num_wires == 4:
        if parameter_sharing == True:
            num_params = 15 * (num_layers + 1)
        else:
             num_params = 60 * num_layers + 15
    elif num_wires == 8:
        if parameter_sharing == True:
            num_params = 15 * (2 * num_layers + 1)
        else:
            num_params = 180 * num_layers + 15
    elif num_wires == 10:
        if parameter_sharing == True:
            num_params = 15 * (2 * num_layers + 2)
        else:
            num_params = 210 * num_layers + 30
    elif num_wires == 12:
        if parameter_sharing == True:
            num_params = 15 * (2 * num_layers + 2)
        else:
            num_params = 270 * num_layers + 30
    return num_params


def get_qcnn(params, num_wires, num_layers, parameter_sharing):
    if num_wires == 4:
        if parameter_sharing is True:
            QCNN4_shared(params, num_layers)
        else:
            QCNN4_not_shared(params, num_layers)
    
    elif num_wires == 8:
        if parameter_sharing is True:
            QCNN8_shared(params, num_layers)
        else:
            QCNN8_not_shared(params, num_layers)

    elif num_wires == 10:
        if parameter_sharing is True:
            QCNN10_shared(params, num_layers)
        else:
            QCNN10_not_shared(params, num_layers)

    elif num_wires == 12:
        if parameter_sharing is True:
            QCNN12_shared(params, num_layers)
        else:
            QCNN12_not_shared(params, num_layers)

#====================================================================================================
# Trace Distance Utility
def get_trace_distance(n_qubits, n_repeats, X_train, Y_train):
    x1 = torch.tensor(X_train[Y_train == 1], dtype=torch.float32)
    x0 = torch.tensor(X_train[Y_train == 0], dtype=torch.float32)

    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(inputs):
        qml.IQPEmbedding(inputs, n_repeats=n_repeats, wires=range(n_qubits))
        return qml.density_matrix(wires=range(n_qubits))

    class distance(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes={})

        def forward(self, x1, x0):
            rhos1 = self.qlayer(x1)
            rhos0 = self.qlayer(x0)

            rho1 = torch.sum(rhos1, dim=0) / len(x1)
            rho0 = torch.sum(rhos0, dim=0) / len(x0)
            rho_diff = rho1 - rho0
            eigvals = torch.linalg.eigvals(rho_diff)
            return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
        
    model = distance()
    return model(x1, x0).item()

def get_trace_distance_TQE(n_qubits, n_repeats, X_train, Y_train, n_layers_TQE=None, trained_params_embedding=None):
    x1 = torch.tensor(X_train[Y_train == 1], dtype=torch.float32)
    x0 = torch.tensor(X_train[Y_train == 0], dtype=torch.float32)

    dev = qml.device("default.qubit", wires=n_qubits)
    def TQE_ansatz(params):
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(n_qubits - 1):
            qml.IsingYY(params[i+n_qubits], wires=[i, i + 1])
        qml.IsingYY(params[2 * n_qubits - 1], wires=[n_qubits - 1, 0])

    @qml.qnode(dev)
    def circuit(inputs):
        for i in range(n_layers_TQE):   
            TQE_ansatz(trained_params_embedding[i * 2 * n_qubits: (i + 1) * 2 * n_qubits])
            qml.IQPEmbedding(inputs, n_repeats=n_repeats, wires=range(n_qubits))
        return qml.density_matrix(wires=range(n_qubits))

    class distance(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes={})

        def forward(self, x1, x0):
            rhos1 = self.qlayer(x1)
            rhos0 = self.qlayer(x0)

            rho1 = torch.sum(rhos1, dim=0) / len(x1)
            rho0 = torch.sum(rhos0, dim=0) / len(x0)
            rho_diff = rho1 - rho0
            eigvals = torch.linalg.eigvals(rho_diff)
            return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
        
    model = distance()
    return model(x1, x0).item()


#====================================================================================================
# NQE Utility
def get_nn_layer(n_qubits, NN_type):
    conv_layer = torch.nn.Sequential(
                # Layer1 : 28 * 28 -> 14 * 14
                torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),

                # Layer2: 14 * 14 -> 7 * 7
                torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),

                torch.nn.Flatten(),

                torch.nn.Linear(7 * 7, 32, bias=True),
                torch.nn.Linear(32, n_qubits, bias=True),)

    linear_relu_stack = torch.nn.Sequential(
                        torch.nn.Linear(n_qubits, 2 * n_qubits),
                        torch.nn.ReLU(),
                        torch.nn.Linear(2 * n_qubits, 4 * n_qubits),
                        torch.nn.ReLU(),
                        torch.nn.Linear(4 * n_qubits, 4 * n_qubits),
                        torch.nn.ReLU(),
                        torch.nn.Linear(4 * n_qubits, 2 * n_qubits),
                        torch.nn.ReLU(),
                        torch.nn.Linear(2 * n_qubits, n_qubits),
                    )

    if NN_type == "conv":
        return conv_layer
    elif NN_type == "linear":
        return linear_relu_stack
    
def optimize_nqe(n_qubits, n_repeats, X, y, nn_type, PATH):
    max_steps = 5000
    dev = qml.device("default.qubit", wires=n_qubits)
        
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
        qml.IQPEmbedding(inputs[0:n_qubits], n_repeats=n_repeats, wires=range(n_qubits))
        qml.adjoint(qml.IQPEmbedding)(inputs[n_qubits:2*n_qubits], n_repeats=n_repeats, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    class NQE_optimize(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
            self.nn_layer = get_nn_layer(n_qubits, nn_type)
        def forward(self, x1, x2):
            x1 = self.nn_layer(x1)
            x2 = self.nn_layer(x2)
            x = torch.concat([x1, x2], 1)
            x = self.qlayer1(x)
            return x[:,0]
            
    def train_models():
        model = NQE_optimize()
        model.train()
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.005)
        loss_history = []
        for it in range(max_steps):
            X1_batch, X2_batch, Y_batch = new_data(64, X, y)
            pred = model(X1_batch, X2_batch)
            pred, Y_batch = pred.to(torch.float32), Y_batch.to(torch.float32)
            loss = loss_fn(pred, Y_batch)
            loss_history.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 1000 == 0:
                print(f"Iterations: {it} Loss: {loss.item()}")
                    
        print("NQE Optimization Complete")
        torch.save(model.state_dict(), PATH + "model.pt")
        return model.state_dict()
        
    train_models()
#====================================================================================================
# score_and_margins Utility
def score_and_margins(model, X, y):
    correct_predictions = 0
    margin_dists = []

    # Convert to tensors
    X_tensor = torch.from_numpy(X).to(torch.float32)
    y_tensor = torch.from_numpy(y).to(torch.long)

    # Get probabilities for the batch
    with torch.no_grad():
        probabilities = model(X_tensor)
            
    # Score calculation
    predictions = torch.argmax(probabilities, dim=1)
    correct_predictions += (predictions == y_tensor).sum().item()
    accuracy = correct_predictions / len(y)

    # Margin calculation
    correct_label_probs = probabilities.gather(1, y_tensor.view(-1, 1)).squeeze(1)
    incorrect_label_probs, _ = torch.max(probabilities.masked_fill(torch.eye(probabilities.size(1))[y_tensor].bool(), float('-inf')), dim=1)
    margin_dist = correct_label_probs - incorrect_label_probs

    # Calculate Margin Mean
    margin_mean = margin_dist.mean().item()
    
    # Zero out margins for incorrect predictions
    margin_dist = torch.where(predictions == y_tensor, margin_dist, torch.zeros_like(margin_dist))
    margin_dists.append(margin_dist)

    # Calculate margin statistics
    margin_dist = torch.cat(margin_dists, dim=0) 
    margin_min = margin_dist.min().item()
    margin_Q1 = torch.quantile(margin_dist, 0.25).item()
    margin_Q2 = torch.quantile(margin_dist, 0.50).item()
    margin_Q3 = torch.quantile(margin_dist, 0.75).item()
    margin_max = margin_dist.max().item()
    margin_boxplot = np.array([margin_min, margin_Q1, margin_Q2, margin_Q3, margin_max])

    return accuracy, margin_dist.detach().numpy(), margin_boxplot, margin_mean