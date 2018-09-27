import json
import os
from collections import deque

prn_json = open(os.path.join(os.path.dirname(__file__), 'prn_info.json')).read()
prn_info = json.loads(prn_json)


class ShiftRegister:
    """
    This class implements a shift register as described in the ICD 200 for using two taps.
    This is used by :class:`PRN`.
    """

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
        Generate the next output and return it. This method includes the feedback step.

        :return: Bit
        """
        out = self.get_output()
        self.do_feedback()
        return out

    def do_feedback(self):
        """
        Generate the feedback, and shift the values.

        :return:
        """
        fb = [self.G[i - 1] for i in self.poly_taps]
        fb = sum(fb) % 2
        self.G.pop()
        self.G.appendleft(fb)

    def get_output(self):
        """
        Generate the next output value for the sequence.

        :return: Bit
        """
        out = [self.G[i - 1] for i in self.prn_taps]
        out = sum(out) % 2
        return out


class PRN:
    """
    This class implements the coarse acquisition prn sequence as described in ICD 200.
    """

    def __init__(self, prn):
        """

        :param prn: SV ID No. as described in the ICD 200.
        """
        sv_range_message = "prn must be 1-63"
        self.prn = prn
        if prn < 1 or prn > 63:
            ValueError(sv_range_message)
        if prn < 38:
            self.sv_prn = prn
        else:
            self.sv_prn = 0
        self.sv_taps = prn_info['taps'][str(self.sv_prn)]
        self.g1 = ShiftRegister(prn_info['taps']['G1'], [10])
        self.g2 = ShiftRegister(prn_info['taps']['G2'], self.sv_taps)
        if prn > 37:
            delays = prn_info['delays'][str(self.prn)]
            delays = list(bin(int(delays, 8))[2:])
            while len(delays) < 10:
                delays.insert(0, 0)
            for i in range(10):
                self.g2.G[i] = int(delays[i])
        self.iteration = 0
        self.ca = []

    def next(self):
        """
        Get the next chip in the sequence.

        :return:
        """
        if self.iteration < 1023:
            g1 = self.g1.next()
            g2 = self.g2.next()
            ca = (g1 + g2) % 2
            self.ca.append(ca)
            self.iteration += 1
        else:
            ca = self.ca[self.iteration % 1023]
            self.iteration += 1
            if self.iteration == 2047:
                self.iteration = 1023
        return ca

    def prn_seq(self):
        """
        Return the full ca sequence. (1023 bits)
        Uses :func:`PRN.next` to generate full sequence.

        :return:
        """
        if self.iteration < 1023:
            while self.iteration < 1023:
                self.next()
        return self.ca