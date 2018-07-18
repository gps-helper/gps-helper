import numpy as np
from numpy.linalg import inv

class SimpleKalman(object):
    """
    Kim Chapter 10.2 Battery voltage estimation with measurement noise

    Python 3.x is assumed so the operator @ can be used for matrix multiply

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