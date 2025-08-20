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
N_TRIALS = 5       # evaluations over n timesteps averaged over n trials
N_TIMESTEPS = int(1e6)   # time steps
EVAL_INTERVAL = 5000  # evaluation interval for the agent

# ========= REINFORCE Hyperparameters ========= #

#####################################
# Hyperparameters - inverted pendulum
#####################################
REINFORCE_TIME_STEPS_WALKER = N_TIMESTEPS
REINFORCE_N_TRIALS = N_TRIALS
HIDDEN_LYR_1_PENDULUM = 256
HIDDEN_LYR_2_PENDULUM = 256
HIDDEN_LYR_3_PENDULUM = 128
LR_PENDULUM = 0.0003
GAMMA_PENDULUM = 0.99    # discount factor on future steps

################################
# Hyperparameters - walker2D
################################
HIDDEN_LYR_1_WALKER = 256
HIDDEN_LYR_2_WALKER = 256
HIDDEN_LYR_3_WALKER = 128
# use a low learning rate because the high variance will cause large gradient updates.
LR_WALKER = 0.0001  # see literature
GAMMA_WALKER = 0.99    # discount factor on future steps
MAX_GRADIENT_NORM = 0.5  # clips the gradient norms for all the policy parameters

# ========= TD3 Hyperparameters ========= #

################################
# Hyperparameters - walker2D
################################

TD3_LR_WALKER = 0.0003
TD3_GAMMA_WALKER = 0.99
TD3_TIME_STEPS = N_TIMESTEPS
TD3_N_TRIALS = N_TRIALS
TD3_REPLAY_BUF_SIZE = TD3_TIME_STEPS
TD3_EVAL_INTERVAL = EVAL_INTERVAL

#####################################
# Hyperparameters - inverted pendulum
#####################################

TD3_LR_PENDULUM = 0.0003
TD3_GAMMA_PENDULUM = 0.99    # discount factor on future steps

################################################################


def train_reinforce(policy,
                    env: gym.Env,
                    obs_dim,
                    action_dim,
                    device='cpu') -> ReinforceTrainer:

    reinforce_agent = ReinforceAgent(
        policy,
        obs_dim,
        action_dim,
        HIDDEN_LYR_1_WALKER,
        HIDDEN_LYR_2_WALKER,
        LR_WALKER,
        GAMMA_WALKER,
        MAX_GRADIENT_NORM,
        device=device
    )

    trainer = ReinforceTrainer(env, reinforce_agent,
                               n_timesteps=REINFORCE_TIME_STEPS_WALKER,
                               n_trials=REINFORCE_N_TRIALS,
                               evaluate_interval=EVAL_INTERVAL)

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


def reinforce_training(gym_sim: str = "Walker2d-v4", device: str = "cpu") -> None:
    """
    Train the Walker2D environment using REINFORCE algorithm.

    """
    # =========== Train REINFORCE Agent ============ #
    sim_env = gym.make(gym_sim)
    obs_dim = sim_env.observation_space.shape[0]
    action_dim = sim_env.action_space.shape[0]

    reinforce_policy = EnhancedPolicyNetwork(
        obs_dim, action_dim,
        HIDDEN_LYR_1_WALKER, HIDDEN_LYR_2_WALKER,
        HIDDEN_LYR_3_WALKER)

    # reinforce_policy = PolicyNetwork(
    #     obs_dim, action_dim, HIDDEN_LYR_1_WALKER, HIDDEN_LYR_2_WALKER)

    reinforce_trainer = train_reinforce(reinforce_policy,
                                        sim_env,
                                        obs_dim,
                                        action_dim,
                                        device=device)

    # =========== Plot Reinforce stats ============ #
    reinforce_av_time_steps, reinforce_mean_returns, reinforce_sds = \
        reinforce_trainer.metrics.get_reinforce_learning()

    learning_curve_plotter.set_reinforce_data(reinforce_av_time_steps,
                                              reinforce_mean_returns,
                                              reinforce_sds)


def td3_training(gym_sim: str = "Walker2d-v4", device: str = "cpu"):
    """
    Main function to train the Walker2D environment using TD3 algorithm.
    """
    # ========== TD3 Training =========== #
    td3_train = train_td3(gym_sim, device)

    td3_average_return, td3_return_std = td3_train.metrics.get_td3_learning()

    td3_x = np.arange(TD3_EVAL_INTERVAL, (len(td3_average_return) + 1) *
                      TD3_EVAL_INTERVAL, TD3_EVAL_INTERVAL)

    learning_curve_plotter.set_td3_data(td3_x,
                                        td3_average_return,
                                        td3_return_std)

#################################################################

# ========== Training Wrapper Functions ======== #


def train_walker_reinforce_v_td3(learning_curve_plotter: PlotLearningCurve,
                                 sim_name="Walker2d-v4",
                                 device: str = "cpu") -> None:
    """
    Train the Walker2D environment using REINFORCE and TD3 algorithms.
    """
    # Train REINFORCE agent
    reinforce_training(sim_name,
                       device)
    # Train TD3 agent
    td3_training(sim_name, device)

    # Plot A/B Learning Curve
    learning_curve_plotter.plot_learning_curves()


#################################################################

if __name__ == "__main__":
    # ========== device setup ========== #
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learning_curve_plotter = PlotLearningCurve(time_steps=TD3_TIME_STEPS,
                                               trials=TD3_N_TRIALS)
    # ========== Training ========== #
    train_walker_reinforce_v_td3(learning_curve_plotter,
                                 sim_name="Walker2d-v4",
                                 device=device)
