import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from src.reinforce.reinforce_agent import ReinforceAgent
from src.reinforce.reinforce_trainer import ReinforceTrainer
from src.TD3.actor_critic_agent import ActorCriticAgent
from src.TD3.td3_trainer import TD3Trainer
from src.util.plotter import learning_rate_ma, evaluation_figure_a_b

##########################################################################

# ========= REINFORCE Hyperparameters ========= #

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
REINFORCE_N_TRIALS = 5
REINFORCE_TIME_STEPS_WALKER = int(100)  # int(1e6)   # time steps
HIDDEN_LYR_1_WALKER = 128
HIDDEN_LYR_2_WALKER = 128
# use a low learning rate because the high variance will cause large gradient updates.
LR_WALKER = 0.0001  # see literature
GAMMA_WALKER = 0.99    # discount factor on future steps
MAX_GRADIENT_NORM = 0.5  # clips the gradient norms for all the policy parameters
REINFORCE_EVAL_INTERVAL = int(5)

# ========= TD3 Hyperparameters ========= #

################################
# Hyperparameters - walker2D
################################

TD3_LR_WALKER = 0.0003
TD3_GAMMA_WALKER = 0.99
TD3_TIME_STEPS = int(1e6)
TD3_N_TRIALS = 10
TD3_REPLAY_BUF_SIZE = TD3_TIME_STEPS
TD3_EVAL_INTERVAL = 5000

################################################################


def train_reinforce(epochs: int,
                    trials: int,
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

    trainer = ReinforceTrainer(sim_env, reinforce_agent,
                               n_timesteps=epochs,
                               n_trials=trials,
                               evaluate_interval=REINFORCE_EVAL_INTERVAL)

    trainer.train()

    return trainer


def train_td3(exp_name: str):
    sim_env = gym.make(exp_name)

    obs_dim = sim_env.observation_space.shape[0]
    action_dim = sim_env.action_space.shape[0]

    td3_actor_critic = ActorCriticAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=TD3_LR_WALKER,
        gamma=TD3_GAMMA_WALKER
    )

    td3_trainer = TD3Trainer(
        sim_env,
        td3_actor_critic,
        time_steps=TD3_TIME_STEPS,
        replay_buffer_size=TD3_REPLAY_BUF_SIZE,
        evaluate_interval=TD3_EVAL_INTERVAL,
        n_trials=TD3_N_TRIALS
    )

    td3_trainer.train()

    return td3_trainer

################################################################

# train_reinforce(EPOCHS, HIDDEN_LYR_1,
#                 HIDDEN_LYR_2, LR,
#                 GAMMA,
#                 'InvertedPendulum-v4')


reinforce_trainer = train_reinforce(REINFORCE_TIME_STEPS_WALKER,
                                    REINFORCE_N_TRIALS,
                                    HIDDEN_LYR_1_WALKER,
                                    HIDDEN_LYR_2_WALKER,
                                    LR_WALKER, GAMMA_WALKER,
                                    "Walker2d-v4",
                                    MAX_GRADIENT_NORM)

# =========== Plot Reinforce stats ============ #

# # h line
# stable_convergence = reinforce_trainer.metrics.stable_convergence_time()


# evaluation_figure_a_b(metrics_a=reinforce_trainer.metrics)

# TODO
# for plotting reinforce learning curve with average returns vs time steps
# because reinforce is on-policy trained per-episode, the time step evaluation (x)
# is provided in a stacked np.array (x,y,z) from the performance metrics method.
# then get arr[:,1] for y axis mean and x[;,0] axis time step values (z is std dev)

reinforce_av_time_steps, reinforce_mean_returns, reinforce_sds = \
    reinforce_trainer.metrics.get_reinforce_learning()

learning_rate_ma(x=reinforce_av_time_steps,
                 y=reinforce_mean_returns,
                 #  target_ep=target_reached,
                 #  convergence_ep=stable_convergence,
                 title=f"Reinforce Learning Curve, {REINFORCE_N_TRIALS} trials. layers: {HIDDEN_LYR_1_WALKER}, {HIDDEN_LYR_2_WALKER}",
                 time_steps=REINFORCE_TIME_STEPS_WALKER
                 )
#################################################################

# ========== TD3 Training =========== #
# td3_train = train_td3("Walker2d-v4")

# td3_average_return, _ = td3_train.metrics.get_average_learning_curve()

# x = np.arange(0, len(td3_average_return)*TD3_EVAL_INTERVAL, TD3_EVAL_INTERVAL)
# learning_rate_ma(x,
#                  y=td3_average_return,
#                  title="TD3 Learning curve, Average over 10 trials")

#################################################################
