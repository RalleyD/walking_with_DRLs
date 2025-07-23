import gymnasium as gym
from src.reinforce.reinforce_agent import ReinforceAgent
from src.reinforce.reinforce_trainer import ReinforceTrainer
from src.util.plotter import learning_rate_ma
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
EPOCHS_WALKER = 10000   # episodes
HIDDEN_LYR_1_WALKER = 64
HIDDEN_LYR_2_WALKER = 64
LR_WALKER = 0.0002
GAMMA_WALKER = 0.99    # discount factor on future steps

################################################################


def train_reinforce(epochs: int,
                    layer_1: int,
                    layer_2: int,
                    lr: float,
                    discount: float,
                    exp_name: str) -> list:

    sim_env = gym.make(exp_name)
    obs_dim = sim_env.observation_space.shape[0]
    action_dim = sim_env.action_space.shape[0]

    reinforce_agent = ReinforceAgent(
        obs_dim, action_dim, layer_1, layer_2, lr, discount)

    trainer = ReinforceTrainer(sim_env, reinforce_agent, epochs)

    return trainer.train()

################################################################

# train_reinforce(EPOCHS, HIDDEN_LYR_1,
#                 HIDDEN_LYR_2, LR,
#                 GAMMA,
#                 'InvertedPendulum-v4')


walker_returns = train_reinforce(EPOCHS_WALKER,
                                 HIDDEN_LYR_1_WALKER,
                                 HIDDEN_LYR_2_WALKER,
                                 LR_WALKER, GAMMA_WALKER,
                                 "Walker2d-v4")

################################################################

x = np.arange(0, len(walker_returns), 1)
y = walker_returns

# fig, ax = plt.subplots(1, 1)
# ax.plot(x, y)

# ax.set_title(
#     f"Reinforce Learning Curve, layers: {HIDDEN_LYR_1_WALKER}, {HIDDEN_LYR_2_WALKER}")
# ax.set_xlabel("episode (n)")
# ax.set_ylabel("returns (n)")

# output_dir = "plots"
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# fig.savefig(output_dir + "/reinforce-lc-walker.png", dpi=600)

# plt.show()
learning_rate_ma(x, np.array(
    y), title=f"Reinforce Learning Curve, layers: {HIDDEN_LYR_1_WALKER}, {HIDDEN_LYR_2_WALKER}")

#################################################################
