#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='Fraunhofer',
      version='1.0.0',
      description='Generic Steller Abundance Determination Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/fraunhofer',
      packages=find_packages(exclude=["tests"]),
      scripts=['bin/hofer'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','doppler @ git+https://github.com/dnidever/doppler.git'],
      #requires=['numpy','astropy(>=4.0)','scipy'],
      #requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils','doppler','synple'],
      include_package_data=True,
)
