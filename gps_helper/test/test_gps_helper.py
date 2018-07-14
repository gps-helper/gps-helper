from .test_helper import GPSTest
from .. import gps_helper as gh
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
tle_path = os.path.abspath(os.path.join(dir_path, '..', '..', 'docs', 'source', 'nb_examples', 'GPS_tle_1_10_2018.txt'))


class TestGPSHelper(GPSTest):
    _multiprocess_can_split_ = True

    def test_GPSDataSource_init(self):
        GPS_ds1 = gh.GPSDataSource(tle_path,
                                    rx_sv_list= \
                                        ('PRN 32', 'PRN 21', 'PRN 10', 'PRN 18'),
                                    ref_lla=(38.8454167, -104.7215556, 1903.0),
                                    ts=1)