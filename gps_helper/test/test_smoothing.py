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
        y_test = np.array([4.07421968, 4.17726944, 4.22942197, 4.33335506, 4.37199144])
        ma = sm.MovingAverageFilter(10)
        Npts = 100
        n = np.arange(0, 100)
        x = 5 * np.cos(2 * np.pi * n / 100)
        v = 0.3 * np.random.randn(Npts)
        z = x + v
        y10 = np.zeros_like(z)
        for k, z_k in enumerate(z):
            y10[k] = ma.next_sample(z_k)
        npt.assert_almost_equal(y10[-5:], y_test)

    def test_moving_average_filter_20(self):
        y_test = np.array([2.87004371, 3.07534742, 3.26104718, 3.41619168, 3.62503732])
        ma = sm.MovingAverageFilter(20)
        Npts = 100
        n = np.arange(0, 100)
        x = 5 * np.cos(2 * np.pi * n / 100)
        v = 0.3 * np.random.randn(Npts)
        z = x + v
        y20 = np.zeros_like(z)
        for k, z_k in enumerate(z):
            y20[k] = ma.next_sample(z_k)
        npt.assert_almost_equal(y20[-5:], y_test)

    def test_low_pass_filter(self):
        y_test = np.array([2.89877566, 3.03393351, 3.15150167, 3.28423681, 3.45673418])
        lf = sm.LowPassFilter(0.9)
        Npts = 100
        n = np.arange(0, 100)
        x = 5 * np.cos(2 * np.pi * n / 100)
        v = 0.3 * np.random.randn(Npts)
        z = x + v
        ylf = np.zeros_like(z)
        for k, z_k in enumerate(z):
            ylf[k] = lf.next_sample(z_k)
        npt.assert_almost_equal(ylf[-5:], y_test)
