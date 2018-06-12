from setuptools import setup
import os
import codecs

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    Warning("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()


def fpath(name):
    return os.path.join(os.path.dirname(__file__), name)


def read(fname):
    return codecs.open(fpath(fname), encoding='utf-8').read()


requirements = read(fpath('requirements.txt'))

setup(name='gps_helper',
      version='0.0.3',
      description='GPS helper module',
      long_description=read_md(fpath('README.md')),
      author='Chiranth Siddappa',
      author_email='chiranthsiddappa@gmail.com',
      url='https://github.com/chiranthsiddappa/gps_helper',
      package_dir={'gps_helper': 'gps_helper'},
      packages=['gps_helper'],
      license='BSD',
      install_requires=requirements.split(),
      test_suite='nose.collector',
      tests_require=['nose', 'tox', 'numpy'],
      )
