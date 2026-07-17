import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch 
from torch import nn, optim
from torchdiffeq import odeint
import optuna 
from tqdm import tqdm

class learn_activation(nn.Module):
    def __init__(self, Iext, J):
        super().__init__()

        self.I_ext = Iext 
        self.J = J

        self.net = None 
        
    def forward(self, t, x):
        return -x + self.net(self.I_ext + self.J * x )
    
def firing_rate_threshold(t, data, T, I_ext, J):
    x = data[0]
    activation = np.maximum(0, I_ext+J*x-T)
    return [-x + activation]

def get_data():
    T_max = 5
    def firing_rate_threshold(t, data, T, I_ext, J):
        x = data[0]
        activation = np.maximum(0, I_ext+J*x-T)
        return [-x + activation]

    sol = solve_ivp(
        firing_rate_threshold, 
        [0, T_max], (x0:=1.,), args=(1, 6, -1), t_eval=np.arange(0, T_max, 0.1),
        rtol = 1e-10, atol=1e-10)

    X_train = sol.y[0][:30:2]
    T_train = sol.t[:30:2]

    X_val = sol.y[0][::3]
    T_val = sol.t[::3]

    x0_torch = torch.tensor([x0], dtype=torch.float32, requires_grad=True)

    t_train_torch = torch.tensor(T_train, dtype=torch.float32)
    x_train_torch = torch.tensor(X_train, dtype=torch.float32).view(-1, 1)
    t_val_torch = torch.tensor(T_val, dtype=torch.float32)
    x_val_torch = torch.tensor(X_val, dtype=torch.float32).view(-1, 1)

    return x0_torch, x_train_torch, t_train_torch, x_val_torch, t_val_torch

def objective(trial):
    x0_torch, x_train_torch, t_train_torch, x_val_torch, t_val_torch = get_data()
    # 2. Suggest values of the hyperparameters using a trial object.
    n_layers = trial.suggest_int('n_layers', 1, 8)
    layers = []

    in_features = 1
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 128)
        activation = trial.suggest_categorical(name=f"activation_l{i}",
            choices=[
                    "ReLU", "GELU", "LeakyReLU", "SiLU", "ELU", "CELU", "SELU", "Mish"
                ]
        )

        activation = getattr(torch.nn, activation)()

        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(activation)
        in_features = out_features

    layers.append(torch.nn.Linear(in_features, 1))
    # layers.append(torch.nn.LogSoftmax(dim=1))

    model = learn_activation(6, -1)
    model.net = nn.Sequential(*layers)
    lr = 0.005
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 1000
    for epoch in (range(epochs)):
        optimizer.zero_grad()
        
        # forward solution
        x_pred = odeint(model, 
                        x0_torch,
                        t_train_torch,
                        method="rk4"
                        )
        
        # Compute loss at training points
        loss = torch.mean((x_pred-x_train_torch)**2)
        
        # Backpropagate
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        x_pred = odeint(model, 
                    x0_torch,
                    t_val_torch,
                    )
        
        loss_val = torch.mean((x_pred-x_val_torch)**2)

    return loss_val

study = optuna.create_study(direction='minimize',
                            study_name="Experiment 2 Followup",
                            storage=f"sqlite:///21_2_optuna_optimization.db",
                            load_if_exists=True)
study.optimize(objective, n_trials=400, n_jobs=8)

# to visualize: optuna-dashboard sqlite:///21_2_optuna_optimization.db 