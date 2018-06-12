from .. import smoothing as sm
from .test_helper import GPSTest
import numpy as np
from numpy import testing as npt


class TestSmoothing(GPSTest):
    _multiprocess_can_split_ = True

    def test_average_filter(self):
        af = sm.RecursiveAverage()
        x_vector = [10, 20, 30]
        for xk in x_vector:
            af.next_sample(xk)
        self.assertEqual(af.x_k, 20)

    def test_average_filter_cos(self):
        y_test = np.array([-0.2327524, -0.18653481, -0.14167614, -0.09500413, -0.04396198])
        Npts = 100
        n = np.arange(0, 100)
        x = 5 * np.cos(2 * np.pi * n / 100)
        v = 0.3 * np.random.randn(Npts)
        z = x + v
        yravg = np.zeros_like(z)
        ravg = sm.RecursiveAverage()
        for k, z_k in enumerate(z):
            yravg[k] = ravg.next_sample(z_k)
        npt.assert_almost_equal(yravg[-5:], y_test)

    def test_moving_average_filter_10(self):
        ma = sm.MovingAverageFilter(10)