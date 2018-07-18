import numpy as np
from numpy.linalg import inv


class SimpleKalman(object):
    """
    Kim Chapter 10.2 Battery voltage estimation with measurement noise

    Python > 3.5 is assumed so the operator @ can be used for matrix multiply

    Mark Wickert February 2018
    """

    def __init__(self, dt=0.2, initial_state=14, P=6):
        """
        Initialize the object
        """
        self.dt = dt
        self.A = np.array([[1]])
        self.H = np.array([[1]])
        # Process model covariance
        self.Q = np.array([[0]])
        # Measurement model covariance
        self.R = 4
        self.x = np.array([[initial_state]])
        self.K = None
        # Error covariance initialize
        self.P = P * np.eye(1)

    def new_sample(self, z):
        """
        Update the Kalman filter state by inputting a new
        scalar measurement. Return the state array as a tuple
        Update all other Kalman filter quantities
        """
        xp = self.A @ self.x
        Pp = self.A @ self.P @ self.A.T + self.Q

        self.K = Pp @ self.H.T * inv(self.H @ Pp @ self.H.T + self.R)

        self.x = xp + self.K @ (np.array([[z]] - self.H @ xp))
        self.P = Pp - self.K @ self.H @ Pp

        self.volt = self.x[0]
        return self.volt


class PosKalman(object):
    """
    Position Estimation from Position and Velocity Measurements

    Python 3.x is assumed so the operator @ can be used for matrix multiply

    Mark Wickert May 2018
    """

    def __init__(self, Q, R, initial_state=[0, 20], dt=0.1):
        """
        Initialize the object
        """
        self.dt = dt
        self.A = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0], [0, 1]])
        # Process model covariance
        self.Q = Q
        # Measurement model covariance
        self.R = R
        self.x = np.array([[initial_state[0]], [initial_state[1]]])
        # Error covariance initialize
        self.P = 5 * np.eye(2)
        # Initialize state
        self.x = np.array([[0.0], [0.0]])

    def new_sample(self, z):
        """
        Update the Kalman filter state by inputting a new
        scalar measurement. Return the state array as a tuple
        Update all other Kalman filter quantities
        """
        xp = self.A @ self.x
        Pp = self.A @ self.P @ self.A.T + self.Q

        self.K = Pp @ self.H.T * inv(self.H @ Pp @ self.H.T + self.R)

        self.x = xp + self.K @ (z - self.H @ xp)
        self.P = Pp - self.K @ self.H @ Pp
        return self.x


class DvKalman(object):
    """
    Kim Chapter 11.2 Velocity from Position Estimation

    Python 3.x is assumed so the operator @ can be used for matrix multiply

    Mark Wickert December 2017
    """

    def __init__(self, initial_state=[0, 20]):
        """
        Initialize the object
        """
        self.dt = 0.1
        self.A = np.array([[1, self.dt], [0, 1]])
        self.H = np.array([[1, 0]])
        # Process model covariance
        self.Q = np.array([[1, 0], [0, 3]])
        # Measurement model covariance
        self.R = 10
        self.x = np.array([[initial_state[0]], [initial_state[1]]])
        # Error covariance initialize
        self.P = 5 * np.eye(2)
        # Initialize pos and vel
        self.pos = 0.0
        self.vel = 0.0

    def new_sample(self, z):
        """
        Update the Kalman filter state by inputting a new
        scalar measurement. Return the state array as a tuple
        Update all other Kalman filter quantities
        """
        xp = self.A @ self.x
        Pp = self.A @ self.P @ self.A.T + self.Q

        self.K = Pp @ self.H.T * inv(self.H @ Pp @ self.H.T + self.R)

        self.x = xp + self.K @ (np.array([[z]] - self.H @ xp))
        self.P = Pp - self.K @ self.H @ Pp

        self.pos = self.x[0]
        self.vel = self.x[1]
        return self.pos, self.vel


class IntKalman(object):
    """
    Kim Chapter 11.4 Position from Velocity Estimation

    Python 3.x is assumed so the operator @ can be used for matrix multiply

    Mark Wickert December 2017
    """

    def __init__(self, initial_state=[0, 20]):
        """
        Initialize the object
        """
        self.dt = 0.1
        self.A = np.array([[1, self.dt], [0, 1]])
        self.H = np.array([[0, 1]])
        # Process model covariance
        self.Q = np.array([[1, 0], [0, 3]])
        # Measurement model covariance
        self.R = 10
        self.x = np.array([[initial_state[0]], [initial_state[1]]])
        # Error covariance initialize
        self.P = 5 * np.eye(2)
        # Initialize pos and vel
        self.pos = 0.0
        self.vel = 0.0

    def new_sample(self, z):
        """
        Update the Kalman filter state by inputting a new scalar measurement.
        Return the state array as a tuple
        Update all other Kalman filter quantities
        """
        xp = self.A @ self.x
        Pp = self.A @ self.P @ self.A.T + self.Q

        self.K = Pp @ self.H.T * inv(self.H @ Pp @ self.H.T + self.R)

        self.x = xp + self.K @ (np.array([[z]] - self.H @ xp))
        self.P = Pp - self.K @ self.H @ Pp

        self.pos = self.x[0]
        self.vel = self.x[1]
        return self.pos, self.vel