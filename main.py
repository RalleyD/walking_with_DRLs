import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from src.reinforce.reinforce_agent import ReinforceAgent
from src.reinforce.reinforce_trainer import ReinforceTrainer
from src.TD3.actor_critic_agent import ActorCriticAgent
from src.TD3.td3_trainer import TD3Trainer
from src.util.plotter import learning_rate_ma, evaluation_figure_a_b, PlotLearningCurve
from src.reinforce.enhanced_policy_network import EnhancedPolicyNetwork
from src.reinforce.policy_network import PolicyNetwork

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
REINFORCE_TIME_STEPS_WALKER = int(1e6)   # time steps
HIDDEN_LYR_1_WALKER = 256
HIDDEN_LYR_2_WALKER = 256
HIDDEN_LYR_3_WALKER = 128
# use a low learning rate because the high variance will cause large gradient updates.
LR_WALKER = 0.0001  # see literature
GAMMA_WALKER = 0.99    # discount factor on future steps
MAX_GRADIENT_NORM = 0.5  # clips the gradient norms for all the policy parameters
REINFORCE_EVAL_INTERVAL = int(5000)

# ========= TD3 Hyperparameters ========= #

################################
# Hyperparameters - walker2D
################################

TD3_LR_WALKER = 0.001
TD3_GAMMA_WALKER = 0.99
TD3_TIME_STEPS = int(1e6)
TD3_N_TRIALS = 5
TD3_REPLAY_BUF_SIZE = TD3_TIME_STEPS
TD3_EVAL_INTERVAL = 5000

################################################################


def train_reinforce(policy,
                    env: gym.Env) -> ReinforceTrainer:

    reinforce_agent = ReinforceAgent(
        policy,
        obs_dim,
        action_dim,
        HIDDEN_LYR_1_WALKER,
        HIDDEN_LYR_2_WALKER,
        LR_WALKER,
        GAMMA_WALKER,
        MAX_GRADIENT_NORM
    )

    trainer = ReinforceTrainer(env, reinforce_agent,
                               n_timesteps=REINFORCE_TIME_STEPS_WALKER,
                               n_trials=REINFORCE_N_TRIALS,
                               evaluate_interval=REINFORCE_EVAL_INTERVAL)

    trainer.train()

    return trainer


def train_td3(exp_name: str, device: str = "cpu") -> TD3Trainer:
    sim_env = gym.make(exp_name)

    # TODO refactor duplication

    obs_dim = sim_env.observation_space.shape[0]
    action_dim = sim_env.action_space.shape[0]

    td3_actor_critic = ActorCriticAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=TD3_LR_WALKER,
        gamma=TD3_GAMMA_WALKER,
        device=device,
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
# ========== device setup ========== #
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO have common timestep and trials constant for A/B
learning_curve_plotter = PlotLearningCurve(time_steps=TD3_TIME_STEPS,
                                           trials=TD3_N_TRIALS)

# =========== Train REINFORCE Agent ============ #
sim_env = gym.make("Walker2d-v4")
obs_dim = sim_env.observation_space.shape[0]
action_dim = sim_env.action_space.shape[0]

reinforce_policy = EnhancedPolicyNetwork(
    obs_dim, action_dim,
    HIDDEN_LYR_1_WALKER, HIDDEN_LYR_2_WALKER,
    HIDDEN_LYR_3_WALKER)

# reinforce_policy = PolicyNetwork(
#     obs_dim, action_dim, HIDDEN_LYR_1_WALKER, HIDDEN_LYR_2_WALKER)

reinforce_trainer = train_reinforce(reinforce_policy,
                                    sim_env)

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

learning_curve_plotter.set_reinforce_data(reinforce_av_time_steps,
                                          reinforce_mean_returns,
                                          reinforce_sds)

# plot a standalone, detailed learning curve
# learning_rate_ma(x=reinforce_av_time_steps,
#                  y=reinforce_mean_returns,
#                  #  target_ep=target_reached,
#                  #  convergence_ep=stable_convergence,
#                  title=f"Reinforce Learning Curve, {REINFORCE_N_TRIALS} trials. layers: {HIDDEN_LYR_1_WALKER}, {HIDDEN_LYR_2_WALKER}",
#                  time_steps=REINFORCE_TIME_STEPS_WALKER,
#                  lyr1=HIDDEN_LYR_1_WALKER,
#                  lyr2=HIDDEN_LYR_2_WALKER,
#                  lyr3=HIDDEN_LYR_3_WALKER
#                  )
#################################################################

# ========== TD3 Training =========== #
td3_train = train_td3("Walker2d-v4", device)

td3_average_return, td3_return_std = td3_train.metrics.get_td3_learning()

td3_x = np.arange(TD3_EVAL_INTERVAL, (len(td3_average_return) + 1) *
                  TD3_EVAL_INTERVAL, TD3_EVAL_INTERVAL)

# plot a standalone detailed learning curve
# learning_rate_ma(x,
#                  y=td3_average_return,
#                  title="TD3 Learning curve, Average over 10 trials")

learning_curve_plotter.set_td3_data(td3_x,
                                    td3_average_return,
                                    td3_return_std)

#################################################################

# ========== Plot A/B Learning Curve =========== #

learning_curve_plotter.plot_learning_curves()

#################################################################
