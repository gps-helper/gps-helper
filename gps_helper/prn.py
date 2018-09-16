import json
import os
from collections import deque

prn_json = open(os.path.join(os.path.dirname(__file__), 'prn_info.json')).read()
prn_info = json.loads(prn_json)


class ShiftRegister:

    def __init__(self, poly_taps, prn_taps):
        """

        :param poly_taps: The polynomial taps to be used for the shift register. Usually G1, or G2 as specified in the
        ICD 200.
        :param prn_taps: The taps for the output, generally 10 for G1, or the sv taps listed in the ICD 200.
        """
        self.G = deque([1 for i in range(10)])  # Needs to be ints for binary addition)
        self.poly_taps = poly_taps
        self.prn_taps = prn_taps

    def next(self):
        """
        Generate the next output and return it. This method includes logic for shifting the bits and binary addition.
        :return: Bit
        """
        out = self.get_output()
        return out

    def do_feedback(self):
        """
        Generate the feedback, and shift the values.
        :return:
        """

    def get_output(self):
        """
        Generate the next output value for the sequence.
        :return: Bit
        """
        out = [self.G[i - 1] for i in self.prn_taps]
        out = sum(out) % 2
        return out