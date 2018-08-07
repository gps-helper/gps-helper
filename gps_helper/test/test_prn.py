from .test_helper import GPSTest
from .. import prn


class TestPRN(GPSTest):
    _multiprocess_can_split_ = True

    def test_prn_info(self):
        prn.prn_info