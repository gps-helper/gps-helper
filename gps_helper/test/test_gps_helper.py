from .test_helper import GPSTest
from .. import gps_helper as gh
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
tle_path = os.path.abspath(os.path.join(dir_path, '..', '..', 'docs', 'source', 'nb_examples', 'GPS_tle_1_10_2018.txt'))


class TestGPSHelper(GPSTest):
    _multiprocess_can_split_ = True

    def setUp(self):
        np.random.seed(31)
        self.gps_ds = gh.GPSDataSource(tle_path,
                                    rx_sv_list= \
                                        ('PRN 32', 'PRN 21', 'PRN 10', 'PRN 18'),
                                    ref_lla=(38.8454167, -104.7215556, 1903.0),
                                    ts=1)

    def test_GPSDataSource_init(self):
        """
        :return:
        """
        self.assertEqual(self.gps_ds.ref_lla, (38.8454167, -104.7215556, 1903.0))
        self.assertEqual(self.gps_ds.GPS_TLE_file, tle_path)
        self.assertEqual(self.gps_ds.Rx_sv_list, ('PRN 32', 'PRN 21', 'PRN 10', 'PRN 18'))
        prn_32 = '1 41328U 16007A   18010.14192069  .00000081  00000-0  00000+0 0  9992',\
                 '2 41328  54.8561 211.7917 0016757 212.5950 147.3462  2.00569342 14112'
        self.assertEqual(self.gps_ds.GPS_sv_dict['PRN 32'], prn_32)
