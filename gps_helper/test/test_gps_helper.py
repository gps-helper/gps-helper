from .test_helper import GPSTest
from .. import gps_helper as gh
import os
import numpy as np
from numpy import testing as npt

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

    def test_GPSDataSource_ll2ecef(self):
        ecef_test = [-1264406.32825878, -4812253.55054197,  3980159.53945133]
        ecef = self.gps_ds.llh2ecef(self.gps_ds.ref_lla)
        npt.assert_almost_equal(ecef, ecef_test)

    def test_GPSDataSource_get_gps_sv(self):
        prn_32_test = '1 41328U 16007A   18010.14192069  .00000081  00000-0  00000+0 0  9992',\
                      '2 41328  54.8561 211.7917 0016757 212.5950 147.3462  2.00569342 14112'
        prn_dict = self.gps_ds.get_gps_sv(tle_path)
        self.assertEqual(prn_dict['PRN 32'], prn_32_test)

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

    def test_GPSDataSource_user_traj_gen_u_enu(self):
        user_enu_test = [[0.001388888888888889, 0.0],
                         [-3.0357660829594124e-18, 0.09999999999999992]]
        rl1 = rl1 = [('e', .2), ('n', .4), ('e', -0.1), ('n', -0.2), ('e', -0.1), ('n', -0.1)]
        user_vel = 5 # mph
        u_pos_enu, u_pos_ecf, sv_pos, sv_vel = self.gps_ds.user_traj_gen(rl1, user_vel,
                                                                         yr2=18,  # the 2k year, so 2018 is 18
                                                                         mon=1,
                                                                         day=15,
                                                                         hr=8 + 7,
                                                                         minute=45)  # Jan 18, 2018, 8:45 AM
        npt.assert_almost_equal(user_enu_test,[[u_pos_enu[0,0], u_pos_enu[0,1]],
                                               (u_pos_enu[-1,0], u_pos_enu[-1,1])])

    def test_GPSDataSource_user_traj_gen_u_ecf(self):
        user_ecef_test = [[-1264404.16643545, -4812254.11855508,  3980159.53945133],
                     [-1264402.00461211, -4812254.68656819,  3980159.53945133],
                     [-1264399.84278877, -4812255.2545813 ,  3980159.53945133],
                     [-1264397.68096543, -4812255.82259441,  3980159.53945133],
                     [-1264395.51914209, -4812256.39060752,  3980159.53945133]]
        rl1 = rl1 = [('e',.2),('n',.4),('e',-0.1),('n',-0.2),('e',-0.1),('n',-0.1)]
        user_vel = 5 # mph
        u_pos_enu, u_pos_ecf, sv_pos, sv_vel = self.gps_ds.user_traj_gen(rl1, user_vel,
                                                                         yr2=18,  # the 2k year, so 2018 is 18
                                                                         mon=1,
                                                                         day=15,
                                                                         hr=8 + 7,
                                                                         minute=45)  # Jan 18, 2018, 8:45 AM
        npt.assert_almost_equal(user_ecef_test, u_pos_ecf[:5])
