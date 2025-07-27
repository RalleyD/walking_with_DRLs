import gymnasium as gym
from src.reinforce.reinforce_agent import ReinforceAgent
from src.reinforce.reinforce_trainer import ReinforceTrainer
from src.util.plotter import learning_rate_ma, evaluation_figure_a_b
import numpy as np
import matplotlib.pyplot as plt
import os

#####################################
# Hyperparameters - inverted pendulum
#####################################
EPOCHS = 2500   # episodes
HIDDEN_LYR_1 = 32
HIDDEN_LYR_2 = 32
LR = 0.0003
GAMMA = 0.99    # discount factor on future steps

################################
# Hyperparameters - walker2D
################################
EPOCHS_WALKER = 200   # episodes
HIDDEN_LYR_1_WALKER = 128
HIDDEN_LYR_2_WALKER = 128
# use a low learning rate because the high variance will cause large gradient updates.
LR_WALKER = 0.000050
GAMMA_WALKER = 0.99    # discount factor on future steps
MAX_GRADIENT_NORM = 0.5  # clips the gradient norms for all the policy parameters
################################################################


def train_reinforce(epochs: int,
                    layer_1: int,
                    layer_2: int,
                    lr: float,
                    discount: float,
                    exp_name: str,
                    grad_clip: float) -> list:

    sim_env = gym.make(exp_name)
    obs_dim = sim_env.observation_space.shape[0]
    action_dim = sim_env.action_space.shape[0]

    reinforce_agent = ReinforceAgent(
        obs_dim, action_dim, layer_1, layer_2, lr, discount, grad_clip)

    trainer = ReinforceTrainer(sim_env, reinforce_agent, epochs)

    return trainer.train(), trainer

################################################################

# train_reinforce(EPOCHS, HIDDEN_LYR_1,
#                 HIDDEN_LYR_2, LR,
#                 GAMMA,
#                 'InvertedPendulum-v4')


walker_returns, reinforce_trainer = train_reinforce(EPOCHS_WALKER,
                                                    HIDDEN_LYR_1_WALKER,
                                                    HIDDEN_LYR_2_WALKER,
                                                    LR_WALKER, GAMMA_WALKER,
                                                    "Walker2d-v4",
                                                    MAX_GRADIENT_NORM)

################################################################

x = np.arange(0, len(walker_returns), 1)
y = walker_returns

# rolling_var, var_coeff = reinforce_trainer.metrics.get_stability_metrics()

# h line
# target_reached = reinforce_trainer.metrics.sample_efficiency()

# h line
# stable_convergence = reinforce_trainer.metrics.stable_convergence_time()


# learning_rate_ma(x, np.array(y),
#                  #  target_ep=target_reached,
#                  convergence_ep=stable_convergence,
#                  title=f"Reinforce Learning Curve, layers: {HIDDEN_LYR_1_WALKER}, {HIDDEN_LYR_2_WALKER}"
#                  )

evaluation_figure_a_b(metrics_a=reinforce_trainer.metrics)

#################################################################
