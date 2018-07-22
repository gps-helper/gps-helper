import numpy as np
from numpy.linalg import inv
from numpy.linalg.linalg import cholesky


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

    def next_sample(self, z):
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

    def next_sample(self, z):
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

    def next_sample(self, z):
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

    def next_sample(self, z):
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


class RadarEKF(object):
    """
    Kim Chapter 14.4 Radar Range Tracking

    Python 3.x is assumed so the operator @ can be used for matrix multiply

    Mark Wickert December 2017
    """

    def __init__(self, dt=0.05, initial_state=[0, 90, 1100]):
        """
        Initialize the object
        """
        self.dt = dt
        self.A = np.eye(3) + self.dt * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        # Process model covariance
        self.Q = np.array([[0, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
        # Measurement model covariance
        self.R = np.array([[10]])
        self.x = np.array(initial_state)
        # Error covariance initialize
        self.P = 10 * np.eye(3)
        # Initialize pos and vel
        self.pos = 0.0
        self.vel = 0.0
        self.alt = 0.0

    def next_sample(self, z):
        """
        Update the Kalman filter state by inputting a new scalar measurement.
        Return the state array as a tuple
        Update all other Kalman filter quantities
        """
        H = self.Hjacob(self.x)
        xp = self.A @ self.x
        Pp = self.A @ self.P @ self.A.T + self.Q

        self.K = Pp @ H.T * inv(H @ Pp @ H.T + self.R)

        self.x = xp + self.K @ (np.array([z - self.hx(xp)]))
        self.P = Pp - self.K @ H @ Pp

        self.pos = self.x[0]
        self.vel = self.x[1]
        self.alt = self.x[2]
        return self.pos, self.vel, self.alt

    def hx(self, xhat):
        """
        State vector predicted to slant range
        """
        zp = np.sqrt(xhat[0] ** 2 + xhat[2] ** 2)
        return zp

    def Hjacob(self, xp):
        """
        Jacobian used to linearize the measurement matrix H
        given the state vector
        """
        H = np.zeros((1, 3))

        H[0, 0] = xp[0] / np.sqrt(xp[0] ** 2 + xp[2] ** 2)
        H[0, 1] = 0
        H[0, 2] = xp[2] / np.sqrt(xp[0] ** 2 + xp[2] ** 2)
        return H

def sigma_points(xm, P, kappa):
    """
    Calculate the Sigma Points of an unscented Kalman filter

    Mark Wickert December 2017
    Translated P. Kim's program from m-code
    """
    n = xm.size
    Xi = np.zeros((n, 2 * n + 1))  # sigma points = col of Xi
    W = np.zeros(2 * n + 1)
    Xi[:, 0, None] = xm
    W[0] = kappa / (n + kappa)

    U = cholesky((n + kappa) * P)  # U'*U = (n+kappa)*P

    for k in range(n):
        Xi[:, k + 1, None] = xm + U[k, None, :].T  # row of U
        W[k + 1] = 1 / (2 * (n + kappa))

    for k in range(n):
        Xi[:, n + k + 1, None] = xm - U[k, None, :].T
        W[n + k + 1] = 1 / (2 * (n + kappa))

    return Xi, W

def ut(Xi, W, noise_cov=0):
    """
    Unscented transformation

    Mark Wickert December 2017
    Translated P. Kim's program from m-code
    """
    n, kmax = Xi.shape

    xm = np.zeros((n, 1))
    for k in range(kmax):
        xm += W[k] * Xi[:, k, None]

    xcov = np.zeros((n, n))
    for k in range(kmax):
        xcov += W[k] * (Xi[:, k, None] - xm) * (Xi[:, k, None] - xm).T

    xcov += noise_cov
    return xm, xcov


class RadarUKF(object):
    """
    Kim Chapter 15.4 Radar Range Tracking UKF Version

    Python 3.x is assumed so the operator @ can be used for matrix multiply

    Mark Wickert December 2017
    """

    def __init__(self, dt=0.05, initial_state=[0, 90, 1100]):
        """
        Initialize the object
        """
        self.dt = dt
        self.n = 3
        self.m = 1
        # Process model covariance
        self.Q = np.array([[0, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
        # Measurement model covariance
        self.R = np.array([[10]])
        self.x = np.array([initial_state]).T
        # Error covariance initialize
        self.P = 100 * np.eye(3)
        self.K = np.zeros((self.n, 1))
        # Initialize pos and vel
        self.pos = 0.0
        self.vel = 0.0
        self.alt = 0.0

    def next_sample(self, z, kappa=0):
        """
        Update the Kalman filter state by inputting a new scalar measurement.
        Return the state array as a tuple
        Update all other Kalman filter quantities
        """
        Xi, W = sigma_points(self.x, self.P, 0)
        fXi = np.zeros((self.n, 2 * self.n + 1))
        for k in range(2 * self.n + 1):
            fXi[:, k, None] = self.fx(Xi[:, k, None])
        xp, Pp = ut(fXi, W, self.Q)

        hXi = np.zeros((self.m, 2 * self.n + 1))
        for k in range(2 * self.n + 1):
            hXi[:, k, None] = self.hx(fXi[:, k, None])
        zp, Pz = ut(hXi, W, self.R)

        Pxz = np.zeros((self.n, self.m))
        for k in range(2 * self.n + 1):
            Pxz += W[k] * (fXi[:, k, None] - xp) @ (hXi[:, k, None] - zp).T

        self.K = Pxz * inv(Pz)
        self.x = xp + self.K * (z - zp)
        self.P = Pp - self.K @ Pz @ self.K.T

        self.pos = self.x[0]
        self.vel = self.x[1]
        self.alt = self.x[2]
        return self.pos, self.vel, self.alt

    def fx(self, x):
        """
        The function f(x) in Kim
        """
        A = np.eye(3) + self.dt * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        xp = A @ x
        return xp

    def hx(self, x):
        """
        The range equation r(x1,x3)
        """
        yp = np.sqrt(x[0] ** 2 + x[2] ** 2)
        return yp