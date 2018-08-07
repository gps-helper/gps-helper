import json
import os

prn_json = open(os.path.join(os.path.dirname(__file__), 'prn_info.json')).read()
prn_info = json.loads(prn_json)


class ShiftRegister:

    def __init__(self, G, poly_taps, prn_taps):
        """

        :param poly_taps: The polynomial taps to be used for the shift register
        :param prn_taps: The taps for the output
        """
        self.G = G
        self.poly_taps = poly_taps
        self.prn_taps = prn_taps
