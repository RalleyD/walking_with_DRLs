# Evaluation Results

discuss, evaluate and interpret the results with reference to the literature



outline the structure of the results chapter - one paragraph



the results shall show how the reserach question was met and answered



use plots and discuss the initial baseline REINFORCE evaluation results

During initial development of the REINFORCE implementation, the network comprised of a simpler structure of two hidden layers of 64 neurons. The raw episode returns were plotted over each episode and a simple moving average overlayed to identify a learning trend:

![Figure: Learning Curve - Initial REINFORCE Model](../plots/Reinforce-Learning-Curve_-layers-64_-64.png)

The learning curve shows high sample variance with a small upward trend, not exceeding an average return of ~300.

Increasing the size of the hidden layers to 128 neurons and the training duration, did not improve the performance of the model:

![Figure: Hyperparameter tuning - REINFORCE - increase model complexity and training duration](../plots/Reinforce-Learning-Curve_-layers-128_-1282025-07-26_03_30_59.png)


use plots and discuss the enhanced baseline REINFORCE policy evaluation results

As devevlopment progressed with the TD3 model. Methods were researched in an attempt to improve the performance and the training stability of REINFORCE; to add fairness to the A/B comparisons:

- The model complexity was increased to the final configuration of three hidden layers (see Methodology) (TODO cite research)
- Xavier initialisation was added to the mean and log-standard-deviation head layers. This attempts to match the variance of the output to the input. (TODO cite)
- Gradient clipping was used to prevent exploding gradients, preventing the loss function swinging between suboptimal solutions (TODO cite).
  - Experiementation showed large gradients (over ~5), leading to high variance in training stability.
  - This was applied to the standard deviation, limiting the size of the value function distributions. The mean distribution output is passed through a tanh activation layer, effectively clamping this output.

![Figure: Model tuning and intialisation enhancements - REINFORCE](../plots/Reinforce-Learning-Curve_-5-trials.-layers-256_-256-lyr1256_-lyr2256_-lyr31282025-08-08_19_41_43.png)

This plot follows the later structure of the training loop. Which averages the evaluations over 5 trials. A trial is a repetition (epoch) of the time steps.

![Figure: enhanced REINFORCE policy - longer training duration](../plots/Reinforce-Learning-Curve_-5-trials.-layers-256_-256-lyr1256_-lyr2256_-lyr31282025-08-09_11_08_00.png)

The enhanced policy shows negligible improvements to the performance, with some improvement to training stability.

An observation to the REINFORCE learning curves, thus far, show that the rate of incremental learning is very slow and almost non-existent. This is likely due to the complexity of the continuous training environment. REINFORCE does not have the capacity to converge to a strong solution over a sufficient period of time. The reward returns achieved by the REINFORCE algorithm are not sufficient to achieve a walking solution.

TODO discuss learning rate tuning, CITE REINFORCE research.


use plots and discuss the initial TD3 results

![Figure: Initial TD3 evaluation - Learning Curve](../plots/TD3-Learning-curve_-Average-over-10-trials2025-08-05_15_20_33.png)

Initial development and evaluation of the TD3 algorithm showed strong returns and provided a stable solution for a walking robot. The networks for the actor and critic follow a 256-256-128 hidden layer structure.

TODO A/B comparison plots

TODO plot and compare the TD3 algorithm with network complexity defined in the Fujimoto paper.

TODO further discuss the results of TD3 hyperparameter turning

TODO discuss limitations

TODO discuss future research work and potential results - decision transformers

TODO Table of results, comparing REINFORCE mean returns, mean S.D, time steps, number of episodes.



