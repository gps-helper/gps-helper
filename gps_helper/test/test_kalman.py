from .test_helper import GPSTest
from .. import kalman as kf
from .. import simulator as sim
import numpy as np
from numpy import testing as npt


class TestKalman(GPSTest):
    """
    Test class for the kalman functions.
    """

    def test_simple_kalman_x(self):
        """
        If only ten measurements are used, the tests do not cover all the way to convergence. Therefore, every tenth
        sample is used for the test.
        :return:
        """
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((len(t), 2))
        x_test = np.array([[13.50229134, 13.50229134],
                           [13.67920055, 13.67920055],
                           [13.35742003, 13.35742003],
                           [13.74166822, 13.74166822],
                           [13.69347514, 13.69347514],
                           [13.81459412, 13.81459412],
                           [13.83609622, 13.83609622],
                           [13.8665808 , 13.8665808 ],
                           [13.77658251, 13.77658251],
                           [13.85373983, 13.85373983],
                           [13.7394069 , 13.7394069 ]])

        # Create objects for the simulation
        gv = sim.GetVoltage(14.0, dt, sigma_w=2)
        sk = kf.SimpleKalman(initial_state=14)

        for k in range(len(t)):
            z = gv.measurement()
            x_saved[k, :] = sk.new_sample(z)
        npt.assert_almost_equal(x_test, x_saved[::10])

    def test_simple_kalman_k(self):
        """
        If only ten measurements are used, the tests do not cover all the way to convergence. Therefore, every tenth
        sample is used for the test.
        :return:
        """
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        k_saved = np.zeros(len(t))
        k_test = np.array([0.6, 0.08571429, 0.04615385, 0.03157895, 0.024,
                           0.01935484, 0.01621622, 0.01395349, 0.0122449 , 0.01090909,
                           0.00983607])

        # Create objects for the simulation
        gv = sim.GetVoltage(14.0, dt, sigma_w=2)
        sk = kf.SimpleKalman(initial_state=14)

        for k in range(len(t)):
            z = gv.measurement()
            sk.new_sample(z)
            k_saved[k] = sk.K
        npt.assert_almost_equal(k_test, k_saved[::10])