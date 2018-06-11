class RecursiveAverage:
    """
    Recursive average filter implementation.
    """

    def __init__(self):
        self.k = 0.0
        self.x_k = 0

    def next_sample(self, x):
        """
        Process the next data sample.

        :param x: Next data sample to compute average
        :return: float
        """
        self.k += 1.0
        self.x_k = ((self.k - 1.) / self.k) * self.x_k + x / self.k
        return self.x_k


class MovingAverageFilter:
    """
    Moving average filter implementation.
    """

    def __init__(self, n):
        """
        :param n: Number of samples to move the average over
        """
        self.n = float(n)
        self.x_k = [0.0 for ii in range(0, int(self.n))]

    def next_sample(self, x):
        """
        Process the next sample.

        :param x: Next data sample to compute average
        :return: float
        """
        self.x_k[1:] = self.x_k[:-1]
        self.x_k[0] = x
        return sum(self.x_k) / self.n


class LowPassFilter:
    """
    Low pass filter implementation.
    """

    def __init__(self, alpha):
        """
        Provide the alpha weighting value to initialize.
        :param alpha: Weighting value
        """
        self.alpha = alpha
        self.prev = 0

    def next_sample(self, x):
        """
        Process the next sample.

        :param x: Next sample value.
        :return: float
        """
        xlpf = self.alpha * self.prev + (1 - self.alpha) * x
        self.prev = xlpf
        return xlpf