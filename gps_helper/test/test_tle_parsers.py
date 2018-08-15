import os
from .test_helper import GPSTest
from .. import tle_parsers as tlep

dir_path = os.path.dirname(os.path.realpath(__file__))
celestrak_tle_path = os.path.abspath(os.path.join(dir_path, '..', '..', 'docs', 'source', 'nb_examples', 'GPS_tle_1_10_2018.txt'))
spacetrack_tle_path = os.path.abspath(os.path.join(dir_path, '..', '..', 'docs', 'source', 'nb_examples', 'Navigation.txt'))


class TestTLEParsers(GPSTest):
    _multiprocess_can_split_ = True

    def test_celestrak(self):
        prn_32_test = '1 41328U 16007A   18010.14192069  .00000081  00000-0  00000+0 0  9992',\
                      '2 41328  54.8561 211.7917 0016757 212.5950 147.3462  2.00569342 14112'
        sv_dict = tlep.get_celestrak_sv(celestrak_tle_path)
        self.assertEqual(sv_dict['PRN 32'], prn_32_test)

    def test_spacetrack(self):
        prn_50_test = '1 26690U 01004A   18224.75202340 -.00000013 +00000-0 +00000-0 0  9993', \
                      '2 26690 053.1312 140.1998 0191738 261.6444 040.2600 02.00577255128475'
        sv_dict = tlep.get_spacetrack_sv(spacetrack_tle_path)
        print(sv_dict)
        self.assertEqual(sv_dict['PRN 50'], prn_50_test)