import logging
import time
import numpy as np
import torch
import pennylane as qml

# Model Training Utility
def train(self, X, y, convergence_interval=10):
    self.model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    loss_history = []
    start = time.time()
    best_train_accuracy = 0.0
    best_model_state = None
    for step in range(self.max_steps):
        
        if self.batch_size == "Full Batch":
            X_batch = torch.tensor(X, dtype=torch.float32)
            y_batch = torch.tensor(y, dtype=torch.long)
        else:
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

        if convergence_interval == "overfit":
            with torch.no_grad():
                train_pred = self.model(torch.tensor(X_batch, dtype=torch.float32))
                train_pred_labels = torch.argmax(train_pred, dim=1)
                train_accuracy = (train_pred_labels == torch.tensor(y_batch)).float().mean().item()
                if train_accuracy >= best_train_accuracy:
                    best_train_accuracy = train_accuracy
                    best_model_state = self.model.state_dict()
            
            if step % 1000 == 0:
                print(f"Step {step}, Loss {loss.item()}, Train Accuracy {train_accuracy}, Best Train Accuracy {best_train_accuracy}")
        else:
            if step % 1000 == 0 :
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
    np.save(self.PATH1 + self.PATH2 + "loss_history.npy", loss_history)
    if convergence_interval == "overfit":
        self.model.load_state_dict(best_model_state)
    for param in self.model.parameters():
        self.weight_final = param.detach().numpy()
   

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