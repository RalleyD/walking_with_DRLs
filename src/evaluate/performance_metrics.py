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

    def __init__(self, ema_alpha=0.5, rolling_window_size=100, convergence_thresh=0.9):
        self._ema_alpha = ema_alpha
        self._rolling_window_size = rolling_window_size
        self._convergence_threshold = convergence_thresh

        # store raw data
        self._episode_returns = []
        self._policy_entropies = []

        # computed metrics
        self._ema_returns = []
        self._best_performance = None
        self._convergence_episode = None

    def update(self, episode_return: float):
        """
        Called after each training episode/epoch.
        Update raw data.
        Compute exponential moving average.
        """
        self._episode_returns.append(episode_return)
        # TODO refactor
        if len(self._ema_returns) == 0:
            self._ema_returns.append(self._episode_returns[-1])
            return

        self._ema_returns.append(
            self._episode_returns[-1] *
            self._ema_alpha + (1 - self._ema_alpha) *
            self._ema_returns[-1]
        )

    def stable_convergence_time(self):
        """
        Find when EMA first reaches and stays above target.
        This two-pointer algorithm determines the longest
        sustained period (n episodes) in the EMA returns list.
        where the EMA stayed above the target threshold.
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

        return convergence_point

    def stability_metrics(sefl):
        """
        Rolling variance and coefficient of variance.
        """
        pass

    def sample_efficiency(self):
        """
        Episodes taken to reach convergance target.
        """
        pass
