import os

dir_path = os.path.dirname(os.path.realpath(__file__))
celestrak_tle_path = os.path.abspath(os.path.join(dir_path, '..', '..', 'docs', 'source', 'nb_examples', 'GPS_tle_1_10_2018.txt'))
spacetrack_tle_path = os.path.abspath(os.path.join(dir_path, '..', '..', 'docs', 'source', 'nb_examples', 'Navigation.txt'))