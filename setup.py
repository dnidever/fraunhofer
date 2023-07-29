#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='Fraunhofer',
      version='1.0.3',
      description='Generic Steller Abundance Determination Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/fraunhofer',
      packages=find_packages(exclude=["tests"]),
      scripts=['bin/hofer'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','thedoppler','annieslasso','synple'],
      #install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','doppler @ git+https://github.com/dnidever/doppler@v1.1.0#egg=doppler',
      #                  'the-cannon @ git+https://github.com/dnidever/AnniesLasso@v1.0.0#egg=the-cannon',
      #                  'synple @ git+https://github.com/dnidever/synple@v1.0.0#egg=synple'],
      #dependency_links=['http://github.com/dnidever/doppler/tarball/v1.1.0#egg=doppler','https://github.com/dnidever/AnniesLasso/tarball/v1.0.0#egg=the-cannon',
      #                  'http://github.com/dnidever/synple/tarball/v1.0.0#egg=synple'],
      include_package_data=True,
)
