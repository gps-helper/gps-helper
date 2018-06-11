from .. import smoothing
from unittest import TestCase

class TestSmoothing(TestCase):

    def test_average_filter(self):
        af = smoothing.RecursiveAverageFilter()
        x_vector = [10, 20, 30]
        for xk in x_vector:
            af.next_sample(xk)
        self.assertEqual(af.x_k, 20)