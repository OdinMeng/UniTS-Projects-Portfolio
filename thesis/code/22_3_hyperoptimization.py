import numpy as np 
from scipy.integrate import solve_ivp
import torch 
from torch import nn, optim
from torchdiffeq import odeint
import optuna 
from tqdm import tqdm
from multiprocessing import Pool
import os 

I0 = 2
J = 10


class learning_winnertakeall_monkeys(nn.Module):
    def __init__(self, I0, J):
        super().__init__()

        self.register_buffer("I0", torch.tensor([I0], dtype=torch.float32))
        self.register_buffer("J", torch.tensor([J], dtype=torch.float32))

        self.net: nn.Module

    def forward(self, t, data):
        activation_r1 = self.net(self.I0 - self.J*data[1]).squeeze()
        activation_r2 = self.net(self.I0 - self.J*data[0]).squeeze()
        out = -data + torch.stack((activation_r1, activation_r2), dim=-1)
        return out

def many_monkeys_simple(t, data, phi, I, J):
    # Assume I0, J are constants
    r1 = data[0]
    r2 = data[1]

    try: 
        I(0)
    except: 
        return np.array([
            -r1 + phi(I - J*r2),
            -r2 + phi(I - J*r1),
        ])
    else:
        return np.array([
            -r1 + phi(I(t) - J*r2),
            -r2 + phi(I(t) - J*r1),
        ])
    
def NP_firing_rate_fun(x):
    return 1 / (1 + np.exp(-x))

def get_data():
    r_init = [0.99, 1.01]
    T_MAX = 12

    sol = solve_ivp(
        many_monkeys_simple,
        t_span = [0, T_MAX],
        y0 = np.array(r_init), 
        t_eval = np.arange(0, T_MAX, 0.01),
        args = (NP_firing_rate_fun, I0, J),
        atol=1e-8, rtol=1e-8
    )

    data_train = (sol.y[:, :551:100]).T
    data_train = np.vstack((data_train, sol.y.T[-400, :]))

    t_train = (sol.t[:551:100])
    t_train = np.append(t_train, sol.t[-400])

    r_init_val = [30, 20]
    T_MAX_val = 1

    sol_val = solve_ivp(
        many_monkeys_simple,
        t_span = [0, T_MAX_val],
        y0 = np.array(r_init_val), 
        t_eval = np.arange(0, T_MAX_val, 0.01),
        args = (NP_firing_rate_fun, I0, J),
        atol=1e-8, rtol=1e-8
    )

    data_val = (sol_val.y[:, ::10]).T
    t_val = (sol_val.t[::10])

    r_init_train_torch = torch.tensor(r_init, dtype=torch.float32)
    r_init_val_torch = torch.tensor(r_init_val, dtype=torch.float32)
    t_train_torch = torch.tensor(t_train, dtype=torch.float32)
    t_val_torch = torch.tensor(t_val, dtype=torch.float32)
    data_train_torch = torch.tensor(data_train, dtype=torch.float32)
    data_val_torch = torch.tensor(data_val, dtype=torch.float32)

    return (r_init_train_torch, r_init_val_torch, t_train_torch, t_val_torch, data_train_torch, data_val_torch)



def objective(trial):
    print(f"Running trial {trial.number=} in process {os.getpid()}")
    torch.set_num_threads(1)
    x0_train_torch, x0_val_torch, t_train_torch, t_val_torch, x_train_torch, x_val_torch = get_data()
    # 2. Suggest values of the hyperparameters using a trial object.
    n_layers = trial.suggest_int('n_layers', 1, 6)
    layers = []

    activation = trial.suggest_categorical(name=f"activations",
        choices=[
                "ReLU", "LeakyReLU", "SiLU", "Tanh", "Sigmoid"
            ]
    )
    activation_f = getattr(torch.nn, activation)()

    in_features = 1
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 128)

        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(activation_f)
        in_features = out_features

    layers.append(torch.nn.Linear(in_features, 1))
    layers.append(activation_f)

    # layers.append(torch.nn.LogSoftmax(dim=1))

    model = learning_winnertakeall_monkeys(I0, J)
    model.net = nn.Sequential(*layers)
    lr = 0.003
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 300

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        # forward solution
        x_pred = odeint(model, 
                        x0_train_torch,
                        t_train_torch,
                        method="dopri5"
                        )
        
        # Compute loss at training points
        loss = torch.mean(torch.norm(x_pred-x_train_torch, p=2, dim=1)**2)
        
        
        # Backpropagate
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        x_pred = odeint(model, 
                    x0_val_torch,
                    t_val_torch,
                    method="dopri5"
                    )
        
        loss_val = torch.mean(torch.norm(x_pred-x_val_torch, p=2, dim=1)**2)

    return loss.item(), loss_val.item()

storage = optuna.storages.RDBStorage(
    url="sqlite:///22_3_optuna_optimization.db",
    engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)

def run_optimization(_):
    study = optuna.create_study(directions=['minimize', 'minimize'],
                                study_name="Experiment 2 Followup",
                                storage=storage,
                                load_if_exists=True,
                                sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=3, n_jobs=1)

if __name__ == "__main__":
    with Pool(processes=12) as pool:
        pool.map(run_optimization, range(12))
    pass 

# to visualize: optuna-dashboard sqlite:///22_3_optuna_optimization.db 