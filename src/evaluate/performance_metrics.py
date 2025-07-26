import numpy as np

from custom_logger import CustomLogger


##############################################
# logger                                     #
##############################################
logger = CustomLogger.get_project_logger()
##############################################


class PerformanceMetrics:
    """
    Maintains and tracks an agent's training loop data.
    Capturing the following metrics:

    Convergence time:
        Time taken to reach a stable e.g 90%
        of the best performance achieved during training.
        This will be measured using exponential moving averages
        to react to recent changes in performance.

    Training stability:
        Rolling variance.
        Coefficient of variance - in order to compare
        stability against models that perform at a higher
        reward level.

    Sample efficiency:
        Episodes needed to reach the stable state e.g 90 %.

    Explotation and Exploration rate:
        Tracking entropy of agent's action distribuitions
        over time.
        Monitoring action variance. High variance indicates
        more exploration.

    Final performance metric:
        Average return in the 90 % stable target.
    """

    def __init__(self, ema_alpha=0.1, rolling_window_size=100, convergence_thresh=0.9):
        self._ema_alpha = ema_alpha
        self._rolling_window_size = rolling_window_size
        self._convergence_threshold = convergence_thresh

        # store raw data
        self._episode_returns = []
        self._policy_entropies = []

        # computed metrics
        self._ema_returns = []
        self._rolling_var = []
        self._rolling_var_coef = []
        self._best_performance = None
        self._convergence_episode = None

    def update(self, episode_return: float, policy_entropy: float = None):
        """
        Called after each training episode/epoch.
        Update raw data.
        Compute exponential moving average.
        """
        self._episode_returns.append(episode_return)

        if policy_entropy:
            self._policy_entropies.append(policy_entropy)

        # TODO refactor
        if not self._ema_returns:
            self._ema_returns.append(self._episode_returns[-1])
            return

        ema = episode_return * \
            self._ema_alpha + (1 - self._ema_alpha) * \
            self._ema_returns[-1]

        self._ema_returns.append(
            ema
        )

        if self._best_performance is None:
            self._best_performance = self._ema_returns[-1]
        else:
            self._best_performance = max(
                self._ema_returns[-1], self._best_performance)

        if len(self._episode_returns)-1 % self._rolling_window_size == 0:
            # update rolling variance
            self._stability_metrics()

    def stable_convergence_time(self):
        """
        Find when EMA first reaches and stays above target.
        This two-pointer algorithm determines the longest
        sustained period (n episodes) in the EMA returns list.
        The method self-determines the stable 'sustained' duration.
        This adds fairness to an agent with high training variance,
        and where different models may reach convergence for different
        durations.
        """
        target = self._best_performance * self._convergence_threshold

        left_ptr = 0
        high_ptr = 0
        convergence_point = 0
        count_a = 0
        count_b = 0

        while left_ptr < len(self._ema_returns)-1:
            if self._ema_returns[left_ptr] < target:
                left_ptr += 1
                high_ptr += 1
            elif self._ema_returns[high_ptr] >= target \
                    and high_ptr < len(self._ema_returns)-1:
                high_ptr += 1
            elif count_a:
                count_b = high_ptr - left_ptr
                if count_b > count_a:
                    count_a = count_b
                    convergence_point = left_ptr
                left_ptr = high_ptr
            else:
                count_a = high_ptr - left_ptr
                convergence_point = left_ptr
                left_ptr = high_ptr

        logger.info(f"\nLongest n_episodes above {target=}: {count_a}")
        logger.info("    Convergence after %d episodes." % convergence_point)

        self._convergence_episode = convergence_point

        return self._convergence_episode

    def _stability_metrics(self):
        """
        Rolling returns variance and coefficient of variance.
        """
        # option 1, perform incrementally from a call to update
        returns_window_np = np.array(
            self._episode_returns[-self._rolling_window_size:])

        var = returns_window_np.var()
        self._rolling_var.append(var)

        var_coeff = np.sqrt(var) / (returns_window_np.mean() +
                                    1e-6)  # avoids zero division error
        self._rolling_var_coef.append(var_coeff)

        # option 2, perform sliding windows on the final episode returns array - slow

    def exploration_trend(self):
        """
        Track how agent exploration changes over time

        This works off of the agent't per-episode average entropy.
        A high entropy indicates the agent is exploring more i.e randomness.
        A low entropy indicates the agent is exploiting more i.e determinism.

        A decaying exploration indicates learning progression.
        Entropy that stays high is a sign of poor convergence or high noise.
        """
        # calculate moving average of entropy
        entropy_ma = 0

        if self._policy_entropies:
            weights = np.ones(self._rolling_window_size) / \
                self._rolling_window_size
            entropy_ma = np.convolve(
                self._policy_entropies, weights, mode='valid')

        return entropy_ma

    def sample_efficiency(self):
        """
        Episodes taken to reach convergance target.
        """
        # TODO refactor duplication
        target = self._best_performance * self._convergence_threshold

        ema_np = np.array(self._ema_returns)
        indices = np.where(ema_np >= target)[0]

        if len(indices):
            sample_eff = int(ema_np[0])
        else:
            sample_eff = 0

        return sample_eff

    def get_stability_metrics(self) -> tuple:
        return self._rolling_var, self._rolling_var_coef
