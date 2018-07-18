import numpy as np


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