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

    def test_PRN_init(self):
        """
        Just testing logic during set up
        :return:
        """
        for sv in range(1, 38):
            prn_seq = prn.PRN(sv)

    def test_first_37(self):
        """
        Test the first ten chips of the first 37 prn's against the provided octal representation from the ICD
        :return:
        """
        for sv in range(1, 38):
            prn_seq = prn.PRN(sv)
            for i in range(10):
                prn_seq.next()
            prn_test = prn.prn_info['first_ten_chips'][str(sv)]
            prn_test = bin(int(prn_test, 8))[2:]
            for i in range(10):
                self.assertEqual(prn_test[i], str(prn_seq.ca[i]))

    def test_rotation(self):
        """
        Test whether the 1024-2047 bits are the same as the 0-1023 bits.
        :return:
        """
        prn_seq = prn.PRN(1)
        ca_first = prn_seq.prn_seq()
        ca_second = []
        for i in range(1023):
            ca_second.append(prn_seq.next())
        self.assertEqual(ca_first, ca_second)

    def test_to_63(self):
        for sv in range(39, 64):
            prn_seq = prn.PRN(sv)
            for i in range(10):
                prn_seq.next()
            prn_test = prn.prn_info['first_ten_chips'][str(sv)]
            prn_test = bin(int(prn_test, 8))[2:]
            prn_test = list(reversed(prn_test))
            while len(prn_test) < 10:
                prn_test.append('0')
            for i in range(10):
                self.assertEqual(prn_test[i], str(prn_seq.ca[i]))