#!/usr/bin/env python

"""SPECFIT.PY - Generic stellar abundance determination software

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200711'  # yyyymmdd

import os
#import sys, traceback
import contextlib, io, sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
from scipy.ndimage.filters import median_filter,gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
import thecannon as tc
from dlnpyutils import utils as dln, bindata
from doppler.spec1d import Spec1D
from doppler import (cannon,utils,reader)
import copy
import emcee
import corner
import logging
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import tempfile
#from . import models
import models
from synple import synple

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


cspeed = 2.99792458e5  # speed of light in km/s

class SpecFitter:
    def __init__ (self,lsf):
        self._teff = []
        self._logg = []
        self._feh = []
        self._lsf = lsf

    def model(self, pars):
        # Create the synthetic spectrum
        synspec = model_spectrum(inputs)
        # Convolve with the LSF
        pspec = prepare_synthspec(synspec,lsf)
        
        return spec


def getabund(inputs):
    """ Grab the abundances out of the input file and return array of abundances."""
    
    # Create the input 99-element abundance array
    codedir = os.path.dirname(os.path.abspath(__file__))
    pertab = Table.read(codedir+'/data/periodic_table.txt',format='ascii')

    feh = inputs.get('feh')
    if feh is None:
        feh = inputs.get('FE_H')
    if feh is None:
        raise ValueError('FE_H missing from inputs')
        
    # Read model atmosphere
    modelfile = inputs.get('modelfile')
    if modelfile is None:
        raise ValueError('modelfile missing from inputs')
    atmostype, teff, logg, vmicro2, mabu, nd, atmos = synple.read_model(modelfile)
    mlines = dln.readlines(modelfile)

    # solar abundances
    # first two are Teff and logg
    # last two are Hydrogen and Helium
    solar_abund = np.array([ 4750., 2.5, 
                            -10.99, -10.66,  -9.34,  -3.61,  -4.21,
                            -3.35,  -7.48,  -4.11,  -5.80,  -4.44,
                            -5.59,  -4.53,  -6.63,  -4.92,  -6.54,
                            -5.64,  -7.01,  -5.70,  -8.89,  -7.09,
                            -8.11,  -6.40,  -6.61,  -4.54,  -7.05,
                            -5.82,  -7.85,  -7.48,  -9.00,  -8.39,
                            -9.74,  -8.70,  -9.50,  -8.79,  -9.52,
                            -9.17,  -9.83,  -9.46, -10.58, -10.16,
                           -20.00, -10.29, -11.13, -10.47, -11.10,
                           -10.33, -11.24, -10.00, -11.03,  -9.86,
                           -10.49,  -9.80, -10.96,  -9.86, -10.94,
                           -10.46, -11.32, -10.62, -20.00, -11.08,
                           -11.52, -10.97, -11.74, -10.94, -11.56,
                           -11.12, -11.94, -11.20, -11.94, -11.19,
                           -12.16, -11.19, -11.78, -10.64, -10.66,
                           -10.42, -11.12, -10.87, -11.14, -10.29,
                           -11.39, -20.00, -20.00, -20.00, -20.00,
                           -20.00, -20.00, -12.02, -20.00, -12.58,
                           -20.00, -20.00, -20.00, -20.00, -20.00,
                           -20.00, -20.00])

    # Scale global metallicity
    abu = solar_abund.copy()
    abu[2:] += feh
    # Now offset the elements with [X/Fe], [X/Fe]=[X/H]-[Fe/H]
    g, = np.where(np.char.array(list(inputs.keys())).find('_H') != -1)
    if len(g)>0:
        ind1,ind2 = dln.match(np.char.array(labels.dtype.names)[g],np.char.array(pertab['symbol']).upper()+'_H')
        abu[ind2] += (np.array(labels[0])[g[ind1]]).astype(float) - feh
    # convert to linear
    abu[2:] = 10**abu[2:]
    # Divide by N(H)
    g, = np.where(np.char.array(mlines).find('ABUNDANCE SCALE') != -1)
    nhtot = np.float64(mlines[g[0]].split()[6])
    abu[2:] /= nhtot
    # use model values for H and He
    abu[0:2] = mabu[0:2]

    return abu


def synple_wrapper(inputs):
    """ This is a wrapper around synple to generate a new synthetic spectrum."""
    # inputs is a dictionary with all of the inputs
    # Teff, logg, [Fe/H], some [X/Fe], and the wavelength parameters (w0, w1, dw).

    # Make the model atmosphere file
    teff = inputs['teff']
    logg = inputs['logg']
    metal = inputs['feh']

    tid,modelfile = tempfile.mkstemp(prefix="mod",dir=".")
    model, header, tail = models.mkmodel(teff,logg,metal,modelfile)
    inputs['modelfile'] = modelfile
    
    # Create the synspec synthetic spectrum
    w0 = inputs['w0']
    w1 = inputs['w1']
    dw = inputs['dw']
    vmicro = inputs.get('vmicro')
    if vmicro is None:
        vmicro = 2.0
    # Get the abundances
    abu = getabund(inputs)
        
    wave,flux,cont = synple.syn(modelfile,(w0,w1),dw,vmicro=vmicro,abu=list(abu))
    
    import pdb; pdb.set_trace()
    
    return spec

def model_spectrum(inputs):
    """
    This creates a model spectrum given the inputs:
    RV, Teff, logg, vmicro, vsini, [Fe/H], [X/Fe], w0, w1, dw.
    This creates the new synthetic spectrum and then convolves with vmicro, vsini and
    shifts to velocity RV.

    """

    # Create the synthetic spectrum
    synspec = synple_wrapper(inputs)

    # 

    return spec

def prepare_synthspec(spec,lsf):
    """ Prepare a synthetic spectrum to be compared to an observed spectrum."""
    # convolve with LSF

    return pspec


def synthspec_jac(x,*args):
     """ Compute the Jacobian matrix (an m-by-n matrix, where element (i, j)
         is the partial derivative of f[i] with respect to x[j]). """

     # A new synthetic spectrum does not need to be generated RV, vmicro or vsini.
     # Some time can be saved by not remaking those.
     # Use a one-sided derivative.
     
     pass

 
def fit(name):
    """ Fit a spectrum and determine the abundances."""

    

        
    # Use curve_fit
    diff_step = np.zeros(npar,float)
    diff_step[:] = 0.02
    lspars, lscov = curve_fit(multispec_interp, wave, flux, sigma=err, p0=initpar, bounds=bounds, jac=synthspec_jac)
    # If it hits a boundary then the solution won't chance much compared to initpar
    # setting absolute_sigma=True gives crazy low lsperror values
    lsperror = np.sqrt(np.diag(lscov))

    if verbose is True:
        print('Least Squares RV and stellar parameters:')
        printpars(lspars)
    lsmodel = multispec_interp(wave,*lspars)
    lschisq = np.sqrt(np.sum(((flux-lsmodel)/err)**2)/len(lsmodel))
    if verbose is True: print('chisq = %5.2f' % lschisq)

    # Put it into the output structure
    dtype = np.dtype([('pars',float,npar),('parerr',float,npar),('parcov',float,(npar,npar)),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['pars'] = lspars
    out['parerr'] = lsperror
    out['parcov'] = lscov
    out['chisq'] = lschisq
