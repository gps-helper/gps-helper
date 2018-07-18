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

    def test_simple_kalman_p(self):
        """
        If only ten measurements are used, the tests do not cover all the way to convergence. Therefore, every tenth
        sample is used for the test.
        :return:
        """
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        p_saved = np.zeros(len(t))
        p_test = np.array([2.4, 0.34285714, 0.18461538, 0.12631579, 0.096,
                           0.07741935, 0.06486486, 0.05581395, 0.04897959, 0.04363636,
                           0.03934426])

        # Create objects for the simulation
        gv = sim.GetVoltage(14.0, dt, sigma_w=2)
        sk = kf.SimpleKalman(initial_state=14)

        for k in range(len(t)):
            z = gv.measurement()
            sk.new_sample(z)
            p_saved[k] = sk.P
        npt.assert_almost_equal(p_test, p_saved[::10])

    def test_pos_kalman_x_pos(self):
        """

        :return:
        """
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((2, len(t)))
        x_test = np.array([2.40104478, 85.65186517, 167.37150554, 248.87502207,
                           328.96915024, 406.71764263, 486.88690622, 566.02913866,
                           645.68324187, 727.69351353, 810.20295338])

        # Create objects for the simulation
        Q = np.array([[1, 0], [0, 3]])
        R = np.array([[10, 0], [0, 2]])
        gpv = sim.GetPosVel(Q=Q, R=R, dt=dt)
        pk = kf.PosKalman(Q, R, initial_state=[0, 80])

        for k in range(len(t)):
            # take a measurement
            z = gpv.measurement()
            # Update the Kalman filter
            x_saved[:, k, None] = pk.new_sample(z)
        npt.assert_almost_equal(x_test, x_saved[0, ::10])

    def test_pos_kalman_x_vel(self):
        """

        :return:
        """
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((2, len(t)))
        x_test = np.array([63.71165764, 81.26369543, 79.46747731, 79.05128724, 77.47045411,
                           78.00121222, 80.21543726, 80.60963484, 81.56581114, 81.90123253,
                           80.31546328])

        # Create objects for the simulation
        Q = np.array([[1, 0], [0, 3]])
        R = np.array([[10, 0], [0, 2]])
        gpv = sim.GetPosVel(Q=Q, R=R, dt=dt)
        pk = kf.PosKalman(Q, R, initial_state=[0, 80])

        for k in range(len(t)):
            # take a measurement
            z = gpv.measurement()
            # Update the Kalman filter
            x_saved[:, k, None] = pk.new_sample(z)
        npt.assert_almost_equal(x_test, x_saved[1, ::10])