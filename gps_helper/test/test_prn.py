from .test_helper import GPSTest
from .. import prn


class TestPRN(GPSTest):
    _multiprocess_can_split_ = True

    def test_prn_info(self):
        prn.prn_info

    def test_prn_G1(self):
        """
        Test the first ten bits of output from G1
        :return:
        """
        g1_test = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        g1_poly = prn.prn_info['taps']['G1']
        sr = prn.ShiftRegister(g1_poly, [10])
        for i in range(3):
            sr.next()
        self.assertEqual(g1_test, list(sr.G))

    def test_prn_G2(self):
        """
        Test the first ten bits of output from G2 using prn 1
        :return:
        """
        g2_test = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        g2_poly = prn.prn_info['taps']['G2']
        prn1_poly = prn.prn_info['taps']['1']
        sr = prn.ShiftRegister(g2_poly, prn1_poly)
        sr.next()
        self.assertEqual(g2_test, list(sr.G))