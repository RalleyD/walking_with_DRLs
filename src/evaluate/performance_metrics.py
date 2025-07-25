

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

    def convergence_time(self):
        """
        Find when EMA first reaches and stays above target.
        """
        target = self._best_performance * self._convergence_threshold

        l = 0
        h = 0
        m = 0
        cnt = 0
        cnt2 = 0

        while l < len(self._ema_returns)-1:
            if self._ema_returns[l] < target:
                l += 1
                h += 1
            elif self._ema_returns[h] >= target and h < len(self._ema_returns)-1:
                h += 1
            elif cnt:
                cnt2 = h - l
                if cnt2 > cnt:
                    cnt = cnt2
                    m = l
                l = h
            else:
                cnt = h - l
                m = l
                l = h

        print(f"longest above thresh {cnt=}")

        return m

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
