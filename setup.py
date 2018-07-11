from setuptools import setup
import os
import codecs


def fpath(name):
    return os.path.join(os.path.dirname(__file__), name)


def read(fname):
    return codecs.open(fpath(fname), encoding='utf-8').read()


requirements = read(fpath('requirements.txt'))
plotting = read(fpath('plotting.txt'))

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gps_helper',
      version='0.0.6',
      description='GPS helper module',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Chiranth Siddappa',
      author_email='chiranthsiddappa@gmail.com',
      url='https://github.com/chiranthsiddappa/gps_helper',
      package_dir={'gps_helper': 'gps_helper'},
      packages=['gps_helper'],
      license='BSD',
      install_requires=requirements.split(),
      test_suite='nose.collector',
      tests_require=['nose', 'tox', 'numpy'],
      extras_require={
            'plotting': plotting.split()
      }
      )
