class AverageFilter:

    def __init__(self):
        self.k = 0.0
        self.x_k = 0

    def next_sample(self, x):
        """

        :param x: Next data sample to compute average
        :return:
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
        self.x_k = [0.0 for ii in range(0, self.n)]

    def next_sample(self, x):
        """

        :param x: Next data sample to compute average
        :return: float
        """
        x_k = self.x_k[0] + (x - self.x_k[-1]) / self.n
        self.x_k[1:] = self.x_k[:-1]
        self.x_k[0] = x_k
        return x_k
