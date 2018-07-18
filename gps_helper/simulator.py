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

