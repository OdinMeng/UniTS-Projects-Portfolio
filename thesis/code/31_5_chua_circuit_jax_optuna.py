# Use optuna to optimize hyperparameters of the NN to learn the CHUA model
import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optax as opx
import matplotlib.pyplot as plt 
import jax.random as jr
import optuna 
from tqdm import tqdm
from multiprocessing import Pool
import os 

class LinearInterpolator(eqx.Module):
    t_obs: jax.Array
    y_obs: jax.Array

    def __call__(self, t):
        t = jnp.clip(t, self.t_obs[0], self.t_obs[-1])
        idx = jnp.searchsorted(self.t_obs, t, side="left")
        idx = jnp.clip(idx, 1, self.t_obs.shape[0] - 1)

        t0, t1 = self.t_obs[idx - 1], self.t_obs[idx]
        y0, y1 = self.y_obs[idx - 1], self.y_obs[idx]
        w = (t - t0) / (t1 - t0 + 1e-12)
        return (1.0 - w) * y0 + w * y1
    
def zero_last_layer(mlp):
    last = mlp.layers[-1]

    new_last = eqx.tree_at(
        lambda l: (l.weight, l.bias),
        last,
        (
            jnp.zeros_like(last.weight),
            jnp.zeros_like(last.bias),
        ),
    )

    mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, new_last)
    return mlp

class ChuaUdeX_PEM(eqx.Module):
    net: eqx.nn.MLP
    alpha: float
    beta: float 
    gamma: float 
    K: jax.Array

    def __init__(self, alpha, beta, gamma, K, key):
        self.net = eqx.nn.MLP(
            in_size='scalar',
            out_size='scalar',
            width_size=64,
            depth=3,
            activation=jax.nn.leaky_relu,
            key=key,
        )
        
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 
        self.K = jnp.asarray(K)

        # zero weight the layer
        self.net = zero_last_layer(self.net)

    def physics(self, t, state):
        # known terms
        x, y, z = state
        return jnp.array(
            [self.alpha*y, x-y+z, -self.beta*y-self.gamma*z]
        )

    def residual(self, t, state):
        x, _, _ = state
        return jnp.array(
            [-self.alpha*self.net(x), 0,0]
        )
    
    def error(self, t, state, obs_fun):
        y_hat = obs_fun(t)
        return y_hat - state

    def __call__(self, t, state, args):
        obs_fun = args
        return self.physics(t, state) + self.residual(t, state) + self.K * self.error(t, state, obs_fun)

def solve_ude_light_pem(model, y0, ts, obs_fun):
    term = dfx.ODETerm(model)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=ts)
    stepper = dfx.PIDController(rtol=1e-7, atol=1e-8)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepper,
        args=obs_fun,
        max_steps=8192
    )
    return sol.ys


@eqx.filter_value_and_grad
def loss_fn_pem(model, x0, y_true, ts, obs_fun):
    # calculate loss
    pred = solve_ude_light_pem(model, x0, ts, obs_fun)
    return jnp.mean((pred - y_true) ** 2)

@eqx.filter_jit
def train_step_pem(model, opt_state, x0, x_true, ts, optimizer, obs_fun):
    # one train step on the dataset
    loss, grads = loss_fn_pem(model, x0, x_true, ts, obs_fun)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def chua_model(t, xyz, args):
    alpha, beta, gamma, a, c = args 
    x,y,z = xyz 
    return jnp.array([
        alpha*(y-a*x**3-c*x),
        x-y+z, 
        -beta*y - gamma*z
    ])

T_MAX = 60 # Constant
ALPHA, BETA, GAMMA, A, C = (10, 16, 0, 1, -0.143) # Constant
@jax.jit
def get_training_set(ALPHA, BETA, GAMMA, A, C):
    term = dfx.ODETerm(chua_model)
    solver = dfx.Dopri5()
    y0 = jnp.array((0.0001, 0.1, 0))
    saveat = dfx.SaveAt(ts=jnp.arange(0, T_MAX, 0.05))

    # training set
    sol_train = dfx.diffeqsolve(term, solver, t0=0, t1=T_MAX, dt0=0.1, y0=y0, args=(ALPHA, BETA, GAMMA, A, C), saveat=saveat)

    x_train = sol_train.ys[::5]
    t_train = sol_train.ts[::5]

    # validation set
    saveat = dfx.SaveAt(ts=jnp.arange(0, T_MAX+T_MAX*0.5, 0.1))
    y0 = jnp.array((0.0001, 0.1, 0.1))
    sol_val = dfx.diffeqsolve(term, solver, t0=0, t1=T_MAX+T_MAX*0.5, dt0=0.1, y0=y0, args=(ALPHA, BETA, GAMMA, A, C), saveat=saveat)

    x_val = sol_val.ys[::2]
    t_val = sol_val.ts[::2]

    return x_train, t_train, x_val, t_val

def objective(trial: optuna.trial.Trial):
    # Step 0: Get data
    x_train, t_train, x_val, t_val = get_training_set(ALPHA, BETA, GAMMA, A, C)

    key = jr.PRNGKey(0)
    linear_interp = LinearInterpolator(t_train, x_train)
    linear_interp_jit = eqx.filter_jit(linear_interp)

    linear_interp_val = LinearInterpolator(t_val, x_val)
    linear_interp_jit_val = eqx.filter_jit(linear_interp_val)

    # K1, K2, K3 between 0 and 1
    Kx = trial.suggest_categorical("K1", [0, 0.1, 0.3, 0.6, 0.9])
    Ky = trial.suggest_categorical("K2", [0, 0.1, 0.3, 0.6, 0.9])
    Kz = trial.suggest_categorical("K3", [0, 0.1, 0.3, 0.6, 0.9])

    ude_pem = ChuaUdeX_PEM(ALPHA, BETA, GAMMA, [float(Kx), float(Ky), float(Kz)], key)

    # Define NN architecture
    activation_fun_str = trial.suggest_categorical('activation', ['relu', 'silu', 'leaky_relu', 'tanh', 'sigmoid'])

    activation_map = {
        'relu': jax.nn.relu,
        'silu': jax.nn.silu,
        'leaky_relu': jax.nn.leaky_relu,
        'tanh': jax.nn.tanh,
        'sigmoid': jax.nn.sigmoid,
    }

    new_net = eqx.nn.MLP(
        in_size='scalar',
        out_size='scalar',
        width_size=64,
        depth=3, 
        activation=activation_map[activation_fun_str],
        key=key
    )

    zeroed_layer = trial.suggest_categorical("zeroed_layer", ['Yes', 'No'])

    if zeroed_layer == 'Yes':
        new_net = zero_last_layer(new_net)

    ude_pem = eqx.tree_at(
        lambda m: m.net,
        ude_pem, 
        new_net
    )
    
    lr = 0.001
    n_epochs = 1000
    optimizer = opx.adam(lr)
    opt_state = optimizer.init(eqx.filter(ude_pem, eqx.is_inexact_array))

    for i in tqdm(range(0, n_epochs)):
        ude_pem, opt_state, train_loss = train_step_pem(ude_pem, opt_state, x_train[0], x_train, t_train, optimizer, linear_interp_jit)

    val_loss = loss_fn_pem(ude_pem, x_val[0], x_val, t_val, linear_interp_jit_val)[0]

    # Return logarithm of the losses for clearer visualization
    return jnp.log10(train_loss), jnp.log10(val_loss)

storage = optuna.storages.RDBStorage(
    url="sqlite:///31_5_chua_optimization.db",
    engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)

def run_optimization(_):
    study = optuna.create_study(directions=['minimize', 'minimize'],
                                study_name="Experiment 2",
                                storage=storage,
                                load_if_exists=True,
                                sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=1, n_jobs=1)

if __name__ == "__main__":
    with Pool(processes=12) as pool:
        pool.map(run_optimization, range(250))
    pass 

# to visualize: optuna-dashboard sqlite:///31_5_chua_optimization.db 