import numpy as np
import os
import warnings
try:
    from scipy.io import loadmat
except ImportError:
    warnings.warn("scipy was not found; used for GetSonar")


class GetVoltage(object):
    """
    A class for generating the battery voltage measurements

    Mark Wickert February 2018
    """

    def __init__(self, batt_voltage=14.4, dt=0.2, sigma_w=2):
        """
        Initialize the object
        """
        self.sigma_w = sigma_w
        self.Voltage_set = batt_voltage

        self.dt = dt

    def measurement(self):
        """
        Take a measurement
        """
        w = 0 + self.sigma_w * np.random.randn(1)[0]
        z = self.Voltage_set + w
        return z


class GetPosVel(object):
    """
    A class for generating position and velocity
    measurements and truth values
    of the state vector.

    Mark Wickert May 2018
    """

    def __init__(self, pos_set=0, vel_set=80.0, dt=0.1,
                 Q=[[1, 0], [0, 3]], R=[[10, 0], [0, 2]]):
        """
        Initialize the object
        """
        self.actual_pos = pos_set
        self.actual_vel = vel_set

        self.Q = np.array(Q)
        self.R = np.array(R)
        self.dt = dt

    def measurement(self):
        """
        Take a measurement
        """
        # Truth position and velocity
        self.actual_vel = self.actual_vel
        self.actual_pos = self.actual_pos \
                          + self.actual_vel * self.dt

        # Measured value is truth plus measurement error
        z1 = self.actual_pos + np.sqrt(self.R[0, 0]) * np.random.randn()
        z2 = self.actual_vel + np.sqrt(self.R[1, 1]) * np.random.randn()
        return np.array([[z1], [z2]])


class GetPos(object):
    """
    A class for generating position measurements as found in Kim

    Mark Wickert December 2017
    """

    def __init__(self, posp=0, vel_set=80.0, dt=0.1,
                 var_w=10.0, var_v=10.0):
        """
        Initialize the object
        """
        self.posp = posp
        self.vel_set = vel_set
        self.velp = vel_set

        self.dt = dt

        self.var_w = var_w
        self.var_v = var_v

    def measurement(self):
        """
        Take a measurement
        """
        # The velocity process noise
        w = 0 + self.var_w * np.random.randn(1)[0]
        # The position measurement noise
        v = 0 + self.var_v * np.random.randn(1)[0]

        # Update the position measurement
        z = self.posp + self.velp * self.dt + v
        # Also update the truth values of position and velocity
        self.posp = z - v
        self.velp = self.vel_set + w
        return z


class GetVel(object):
    """
    A class for generating velocity measurements as found in Kim 11.4

    Mark Wickert December 2017
    """

    def __init__(self, Pos_set=0, Vel_set=80.0, dt=0.1, var_v=10.0):
        """
        Initialize the object
        """
        self.Posp = Pos_set
        self.Vel_set = Vel_set
        self.Velp = Vel_set

        self.dt = dt

        self.var_v = var_v

    def measurement(self):
        """
        Take a measurement
        """
        # The velocity measurement noise
        v = 0 + self.var_v * np.random.randn(1)[0]

        # Also update the truth values of position and velocity
        self.Posp += self.Velp * self.dt
        self.Velp = self.Vel_set + v
        z = self.Velp
        return z


class GetSonar(object):
    """
    A class for playing back sonar altitude measurements as found in Kim 2.4
    and later used in Kim 11.5

    This example requires the scipy package to load a .mat file.

    Mark Wickert December 2017
    """

    def __init__(self):
        """
        Initialize the object
        """
        sonar_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'source', 'nb_examples', 'SonarAlt.mat')
        sonar_file = open(sonar_path)
        sonarD = loadmat(sonar_file)
        self.h = sonarD['sonarAlt'].flatten()
        self.Max_pts = len(self.h)
        self.k = 0

    def measurement(self):
        """
        Take a measurement
        """

        h = self.h[self.k]
        self.k += 1
        if self.k > self.Max_pts:
            print('Recycling data by starting over')
            self.k = 0
        return h