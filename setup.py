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
      #install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','doppler @ git+https://github.com/dnidever/doppler@master#egg=doppler',
      #                  'thecannon @ git+http://github.com/andycasey/AnniesLasso@master#egg=thecannon'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','doppler @ git+https://github.com/dnidever/doppler@v1.1.0#egg=doppler',
                        'the-cannon @ git+https://github.com/sdss/thecannon@1.0.0#egg=the-cannon'],
      $                  'the-cannon @ https://github.com/andycasey/AnniesLasso/archive/master.zip'],      
      #                  'the-cannon @ git+http://github.com/andycasey/AnniesLasso@master#egg=the-cannon'],
      #                  'thecannon @ https://github.com/andycasey/AnniesLasso/archive/master.zip'],
      #                  'thecannon'],
      #requires=['numpy','astropy(>=4.0)','scipy'],
      #requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils','doppler','synple'],
      #dependency_links=['http://github.com/dnidever/doppler/tarball/master#egg=doppler','http://github.com/andycasey/AnniesLasso/tarball/master#egg=thecannon'],
      dependency_links=['http://github.com/dnidever/doppler/tarball/v1.1.0#egg=doppler','https://github.com/sdss/thecannon@1.0.0#egg=the-cannon'],
      include_package_data=True,
)
