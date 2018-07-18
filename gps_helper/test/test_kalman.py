from .test_helper import GPSTest
from .. import kalman as kf
from .. import simulator as sim
import numpy as np
from numpy.linalg import norm
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
        If only ten measurements are used, the tests do not cover all the way to convergence. Therefore, every tenth
        sample is used for the test.
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
        If only ten measurements are used, the tests do not cover all the way to convergence. Therefore, every tenth
        sample is used for the test.
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

    def test_pos_kalman_p_pos(self):
        """
        If only ten measurements are used, the tests do not cover all the way to convergence. Therefore, every tenth
        sample is used for the test.
        :return:
        """
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((2, len(t)))
        p_diag = np.zeros((len(t),2))
        p_test = np.array([3.76669267, 2.76774029, 2.76843135, 2.7685862, 2.76859725,
                           2.76859802, 2.76859808, 2.76859808, 2.76859808, 2.76859808,
                           2.76859808])

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
            p_diag[k, :] = pk.P.diagonal()
        npt.assert_almost_equal(p_test, p_diag[::10, 0])

    def test_pos_kalman_p_vel(self):
        """
        If only ten measurements are used, the tests do not cover all the way to convergence. Therefore, every tenth
        sample is used for the test.
        :return:
        """
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((2, len(t)))
        p_diag = np.zeros((len(t),2))
        p_test = np.array([1.59079563, 1.36849424, 1.36839693, 1.36838998, 1.3683895 ,
                           1.36838946, 1.36838946, 1.36838946, 1.36838946, 1.36838946,
                           1.36838946])

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
            p_diag[k, :] = pk.P.diagonal()
        npt.assert_almost_equal(p_test, p_diag[::10, 1])

    def test_dv_kalman_x_pos(self):
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((len(t), 2))
        x_test = np.array([3.00505891, 81.24273033, 159.15363574, 239.96205645,
                           306.83657799, 390.95995075, 482.18752904, 560.48756223,
                           638.8627147, 716.34708371, 796.09003321])

        # Create objects for the simulation
        gp = sim.GetPos()
        dk = kf.DvKalman()

        for k in range(len(t)):
            z = gp.measurement()

            x_saved[k, :] = dk.new_sample(z)
        npt.assert_almost_equal(x_test, x_saved[::10, 0])

    def test_dv_kalman_x_vel(self):
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((len(t), 2))
        x_test = np.array([20.08306272, 59.87779911, 72.3189212 , 73.53420293, 64.12422919,
                           66.87463733, 84.32133029, 79.23214373, 82.47667741, 86.30916162,
                           80.29449928])

        # Create objects for the simulation
        gp = sim.GetPos()
        dk = kf.DvKalman()

        for k in range(len(t)):
            z = gp.measurement()

            x_saved[k, :] = dk.new_sample(z)
        npt.assert_almost_equal(x_test, x_saved[::10, 1])

    def test_int_kalman_x_vel(self):
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((len(t), 2))
        x_test = np.array([44.82330127, 80.66266235, 72.73385292, 82.05528032, 74.8722922 ,
                           82.66549625, 86.53350242, 80.31837348, 80.74624911, 83.77797869,
                           80.57758014])

        # Create objects for the simulation
        gv = sim.GetVel()
        ik = kf.IntKalman()

        for k in range(len(t)):
            z = gv.measurement()

            x_saved[k, :] = ik.new_sample(z)
        npt.assert_almost_equal(x_test, x_saved[::10, 1])

    def test_int_kalman_x_pos(self):
        dt = 0.1
        t = np.arange(0, 10 + dt, dt)
        x_saved = np.zeros((len(t), 2))
        x_test = np.array([3.55145633, 77.7381104 , 153.44154205, 235.38042071,
                           313.80303816, 394.62000052, 473.96915187, 554.86351,
                           630.47869611, 712.59481293, 786.37162823])

        # Create objects for the simulation
        gv = sim.GetVel()
        ik = kf.IntKalman()

        for k in range(len(t)):
            z = gv.measurement()

            x_saved[k, :] = ik.new_sample(z)
        npt.assert_almost_equal(x_test, x_saved[::10, 0])

    def test_ekf_x(self):
        dt = 0.05
        n_samples = 500
        t = np.arange(n_samples) * dt
        n_samples = len(t)
        x_saved = np.zeros((n_samples, 3))
        x_test = np.array([[   4.5       ,   90.        , 1048.35700227],
                           [ 340.92701179,  129.19059199, 1007.64740711],
                           [ 793.01850282,  153.83039049, 1003.76887912],
                           [1187.81309238,  156.6880102 , 1003.96499988],
                           [1635.12480478,  169.35387149, 1006.01391596],
                           [1965.96084856,  148.35186341, 1000.27827996],
                           [2411.82207964,  162.38496029, 1004.06328925],
                           [2811.82330229,  161.8190544 , 1003.97565162],
                           [3254.16219351,  168.74746912, 1004.5385335 ],
                           [3630.89648321,  154.58882883, 1004.22494179]])

        gr = sim.GetRadar()
        ekf = kf.RadarEKF(dt, initial_state=[0, 90, 1100])
        for k in range(n_samples):
            xm = gr.measurement()
            x_saved[k, :] = ekf.new_sample(xm)
        npt.assert_almost_equal(x_test, x_saved[::50, :])

    def test_ekf_z(self):
        dt = 0.05
        n_samples = 500
        t = np.arange(n_samples) * dt
        n_samples = len(t)
        x_saved = np.zeros((n_samples, 3))
        z_saved = np.zeros(n_samples)
        z_test = np.array([1052.22272082, 1071.57581789, 1288.44638908, 1563.13684492,
       1927.27212923, 2210.78424339, 2617.51747043, 2990.06735689,
       3409.85996124, 3770.3812422 ])

        gr = sim.GetRadar()
        ekf = kf.RadarEKF(dt, initial_state=[0, 90, 1100])
        for k in range(n_samples):
            xm = gr.measurement()
            x_saved[k, :] = ekf.new_sample(xm)
            z_saved[k] = norm(x_saved[k])
        npt.assert_almost_equal(z_test, z_saved[::50])