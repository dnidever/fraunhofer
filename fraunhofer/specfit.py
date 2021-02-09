#!/usr/bin/env python

"""SPECFIT.PY - Generic stellar abundance determination software

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200711'  # yyyymmdd

import os
import shutil
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
import doppler
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
    def __init__ (self,spec,allparams):
        self.allparams = allparams
        self.fitparams = list(allparams.keys())  # by default fit all parameters
        self.lsf = spec.lsf.copy()
        # Figure out the wavelength parameters
        npix,norder = spec.flux.shape
        xp = np.arange(npix//20)*20
        wr = np.zeros((spec.lsf.norder,2),np.float64)
        dw = np.zeros(spec.lsf.norder,np.float64)
        mindw = np.zeros(norder,np.float64)
        for o in range(spec.norder):
            dw[o] = np.median(dln.slope(spec.wave[:,o]))
            wr[o,0] = np.min(spec.wave[:,o])
            wr[o,1] = np.max(spec.wave[:,o])            
            fwhm = spec.lsf.fwhm(spec.wave[xp,o],xtype='Wave',order=o)
            # FWHM is in units of lsf.xtype, convert to wavelength/angstroms, if necessary
            if spec.lsf.xtype.lower().find('pix')>-1:
                fwhm *= np.abs(dw[o])

            # need at least ~4 pixels per LSF FWHM across the spectrum
            #  using 3 affects the final profile shape
            mindw[o] = np.min(fwhm/4)
        self._dw = np.min(mindw)
        self._w0 = np.min(spec.wave)
        self._w1 = np.max(spec.wave)

    @property
    def allparams(self):
        return self._allparams

    @allparams.setter
    def allparams(self,allparams):
        """ Dictionary, keys must be all CAPS."""
        self._allparams = dict((key.upper(), value) for (key, value) in allparams.items())  # all CAPS

    @property
    def fitparams(self):
        return self._fitparams

    @fitparams.setter
    def fitparams(self,fitparams):
        """ list, keys must be all CAPS."""
        self._fitparams = [v.upper() for v in fitparams]      # all CAPS
        
    def mkinputs(self,args):
        """ Make INPUTS dictionary."""
        # Create INPUTS with all arguments needed to make the spectrum
        inputs = self.allparams.copy()  # initialize with initial/fixed values
        for k in range(len(self.fitparams)):        # this overwrites the values for the fitted values
            inputs[self.fitparams[k]] = args[k]
        inputs['DW'] = self._dw          # add in wavelength parameters
        inputs['W0'] = self._w0
        inputs['W1'] = self._w1
        return inputs
        
    def model(self, xx, *args):
        print(args)
        # The arguments correspond to the fitting parameters
        inputs = self.mkinputs(args)
        ## Create INPUTS with all arguments needed to make the spectrum
        #inputs = self.allparams.copy()  # initialize with initial/fixed values
        #for k in range(len(self.fitparams)):        # this overwrites the values for the fitted values
        #    inputs[self.fitparams[k]] = args[k]
        #inputs['dw'] = self._dw          # add in wavelength parameters
        #inputs['w0'] = self._w0
        #inputs['w1'] = self._w1
        print(inputs)
        # Create the synthetic spectrum
        synspec = model_spectrum(inputs)
        # Convolve with the LSF
        pspec = prepare_synthspec(synspec,self.lsf)
        # Return flattened spectrum
        
        return pspec.flux.flatten()

    def jac(self,x,*args,verbose=False):
        """ Compute the Jacobian matrix (an m-by-n matrix, where element (i, j)
        is the partial derivative of f[i] with respect to x[j]). """

        # A new synthetic spectrum does not need to be generated RV, vmicro or vsini.
        # Some time can be saved by not remaking those.
        # Use a one-sided derivative.

        relstep = 0.02
        npix = len(x)
        npar = len(args)

        # Get INPUTS dictionary and make keys all CAPS
        inputs = self.mkinputs(args)
        inputs = dict((key.upper(), value) for (key, value) in inputs.items())

        # Extend on the ends for RV/convolution purposes
        w0 = inputs['W0']
        w1 = inputs['W1']
        dw = inputs['DW']
        rv = inputs.get('RV')
        vrot = inputs.get('VROT')
        vmicro = inputs.get('VMICRO')        
        inputsext = inputs.copy()
        if rv is not None or vrot is not None:
            numext = int(np.ceil(w1*(1.0+1500/cspeed)-w1))
            inputsext['W0'] = w0-numext*dw
            inputsext['W1'] = w1+numext*dw

        # Create synthetic spectrum at current values
        #  set vrot=vmicro=0, will convolve later if necessary
        inputsext['VMICRO'] = 0
        inputsext['VROT'] = 0
        wave1,flux1,cont1 = synple_wrapper(inputsext,verbose=verbose)
        # Get final wavelength array
        wv1, ind1 = dln.closest(wave1,w0)
        wv2, ind2 = dln.closest(wave1,w1)
        origspec = Spec1D(flux1/cont1,wave=wave1,lsfpars=np.array(0.0))
        origspec.cont = cont1
        # Smooth and shift
        if rv is not None or vrot is not None or vmicro is not None:
            smorigspec = smoothshift_spectrum(origspec,vrot=vrot,vmicro=vmicro,rv=rv)
        else:
            smorigspec = origspec.copy()
        # Trim to final wavelengths
        if rv is not None or vrot is not None:
            temp = smorigspec.copy()
            smorigspec = Spec1D(temp.flux[ind1:ind2+1],wave=temp.wave[ind1:ind2+1],lsfpars=np.array(0.0))
            if hasattr(temp,'cont'):
                smorigspec.cot = temp.cont[ind1:ind2+1]
            del temp
        # Convolve with the LSF
        pspec = prepare_synthspec(smorigspec,self.lsf)
        # Flatten the spectrum
        origflux = pspec.flux.flatten()


        # IS IT FASTER TO SMOOTH ALL OF THE SPECTRA THE SAME TIME??
        
        
        # Initialize jacobian matrix
        jac = np.zeros((npix,npar),np.float64)
        # Model at current values
        f0 = multispec_interp(x,*args)


        
        # Compute full models for teff/logg/feh
        for i in range(3):
            pars = np.array(copy.deepcopy(argv))
            step = relstep*pars[i]
            pars[i] += step
            f1 = multispec_interp(x,*pars)
            # Hit an edge, try the negative value instead
            nbd = np.sum(f1>1000)
            if nbd>1000:
                pars = np.array(copy.deepcopy(argv))
                step = -relstep*pars[i]
                pars[i] += step
                f1 = multispec_interp(x,*pars)
            jac[:,i] = (f1-f0)/step
        # Compute model for single spectra
        nspec = len(speclist)
        cnt = 0
        for i in range(nspec):
            vrel1 = vrel[i]
            step = 1.0
            vrel1 += step
            npx = speclist[i].npix*speclist[i].norder
            m = modlist[i]([teff,logg,feh],rv=vrel1)
            if m is not None:
                jac[cnt:cnt+npx,i] = (m.flux.T.flatten()-f0[cnt:cnt+npx])/step
            else:
                jac[cnt:cnt+npx,i] = 1e30
            cnt += npx
                
        return jac



def getabund(inputs,verbose=False):
    """ Grab the abundances out of the input file and return array of abundances."""
    
    # Create the input 99-element abundance array
    codedir = os.path.dirname(os.path.abspath(__file__))
    pertab = Table.read(codedir+'/data/periodic_table.txt',format='ascii')

    feh = inputs.get('FEH')
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
    g, = np.where( (np.char.array(list(inputs.keys())).find('_H') != -1) &
                   (np.char.array(list(inputs.keys())) != 'FE_H') )
    if len(g)>0:
        ind1,ind2 = dln.match(np.char.array(list(inputs.keys()))[g],np.char.array(pertab['symbol']).upper()+'_H')
        for k in range(len(ind1)):
            key1 = np.char.array(list(inputs.keys()))[g[ind1[k]]]
            abu[ind2[k]] += float(inputs[key1]) - feh
            if verbose:
                print('%s %f' % (key1,float(inputs[key1])))
    # convert to linear
    abu[2:] = 10**abu[2:]
    # Divide by N(H)
    g, = np.where(np.char.array(mlines).find('ABUNDANCE SCALE') != -1)
    nhtot = np.float64(mlines[g[0]].split()[6])
    abu[2:] /= nhtot
    # use model values for H and He
    abu[0:2] = mabu[0:2]

    return abu


def synple_wrapper(inputs,verbose=False,tmpbase='/tmp'):
    """ This is a wrapper around synple to generate a new synthetic spectrum."""
    # inputs is a dictionary with all of the inputs
    # Teff, logg, [Fe/H], some [X/Fe], and the wavelength parameters (w0, w1, dw).

    # Make temporary directory for synple to work in
    curdir = os.path.abspath(os.curdir) 
    tdir = os.path.abspath(tempfile.mkdtemp(prefix="syn",dir=tmpbase))
    os.chdir(tdir)
    
    # Make key names all CAPS
    inputs = dict((key.upper(), value) for (key, value) in inputs.items())
    
    # Make the model atmosphere file
    teff = inputs['TEFF']
    logg = inputs['LOGG']
    metal = inputs['FE_H']

    tid,modelfile = tempfile.mkstemp(prefix="mod",dir=".")
    model, header, tail = models.mkmodel(teff,logg,metal,modelfile)
    inputs['modelfile'] = modelfile
    
    # Create the synspec synthetic spectrum
    w0 = inputs['W0']
    w1 = inputs['W1']
    dw = inputs['DW']
    vmicro = inputs.get('VMICRO')
    vrot = inputs.get('VROT')
    if vrot is None:
        vrot = 0.0
    # Get the abundances
    abu = getabund(inputs,verbose=verbose)
    
    wave,flux,cont = synple.syn(modelfile,(w0,w1),dw,vmicro=vmicro,vrot=vrot,abu=list(abu))

    # Delete temporary files
    shutil.rmtree(tdir)
    os.chdir(curdir)
    
    return (wave,flux,cont)


def smoothshift_spectrum(inpspec,vmicro=None,vrot=None,rv=None):
    """ This smoothes the spectrum by Vrot+Vmicro and
        shifts it by RV."""

    #vmicro = inputs.get('VMICRO')
    #vrot = inputs.get('VROT')
    #rv = inputs.get('RV')

    # Nothing to do
    if vmicro is None and vrot is None and rv is None:
        return inpspec.copy()
    
    # Initialize output spectrum
    spec = inpspec.copy()

    # Some broadening
    if vmicro is not None or vrot is not None:
        flux = doppler.utils.broaden(spec.wave,spec.flux,vgauss=vmicro,vsini=vrot)
        spec.flux = flux
            
    ## Vrot/Vsini (km/s) and Vmicro (in km/s)
    #if vrot is not None or vmicro is not None:
    #    wave, flux = synple.call_rotin(wave, flux, vrot, fwhm, space, steprot, stepfwhm, clean=False, reuseinputfiles=True)
        
    # Doppler shift only (in km/s)
    if rv is not None:
        if rv != 0.0:
            shiftwave = spec.wave*(1+rv/cspeed)
            gd,ngd,bd,nbd = dln.where( (spec.wave >= np.min(shiftwave)) & (spec.wave <= np.max(shiftwave)), comp=True)
            # Doppler shift and interpolate onto wavelength array
            if hasattr(spec,'cont'):
                cont = synple.interp_spl(spec.wave[gd], shiftwave, spec.cont)
                spec.cont *= 0
                spec.cont[gd] = cont
                # interpolate the continuing to the missing pixels
                if nbd>0:
                    contmissing = dln.interp(spec.wave[gd],spec.cont[gd],spec.wave[bd],kind='linear',assume_sorted=False)
                    spec.cont[bd] = contmissing
            flux = synple.interp_spl(spec.wave[gd], shiftwave, spec.flux)
            spec.flux *= 0
            spec.flux[gd] = flux
            if nbd>0:
                # Fill in missing values with interpolated values
                coef = dln.poly_fit(spec.wave[gd],spec.flux[gd],2)
                fluxmissing = dln.poly(spec.wave[bd],coef)
                spec.flux[bd] = fluxmissing
                # Mask these pixels
                if spec.mask is None:
                    spec.mask = np.zeros(len(spec.flux),bool)
                spec.mask[bd] = True
    
    return spec


def model_spectrum(inputs,verbose=False,keepextend=False):
    """
    This creates a model spectrum given the inputs:
    RV, Teff, logg, vmicro, vsini, [Fe/H], [X/Fe], w0, w1, dw.
    This creates the new synthetic spectrum and then convolves with vmicro, vsini and
    shifts to velocity RV.

    """
    
    # Make key names all CAPS
    inputs = dict((key.upper(), value) for (key, value) in inputs.items())

    # Extend on the ends for RV/convolution purposes
    w0 = inputs['W0']
    w1 = inputs['W1']
    dw = inputs['DW']
    rv = inputs.get('RV')
    inputsext = inputs.copy()
    if rv is not None:
        numext = int(np.ceil(w1*(1.0+1500/cspeed)-w1))
        inputsext['W0'] = w0-numext*dw
        inputsext['W1'] = w1+numext*dw
        if verbose:
            print('Extending wavelength by '+str(numext)+' pixels on each end')
        
    # Create the synthetic spectrum
    wave1,flux1,cont1 = synple_wrapper(inputsext,verbose=verbose)

    # Vmicro, handled by synple already
    # Vsini, handled by synple as VROT
    # Doppler shift
    if rv is not None:
        # Get final wavelength array
        wv1, ind1 = dln.closest(wave1,w0)
        wv2, ind2 = dln.closest(wave1,w1)
        wave = wave1[ind1:ind2+1]
        # Doppler shift and interpolate onto final wavelength array
        cont = synple.interp_spl(wave, wave1*(1+rv/cspeed), cont1)
        flux = synple.interp_spl(wave, wave1*(1+rv/cspeed), flux1)        
    
    # gaussian convolution, synple.lgconv()
    # rotation broadening, synple.rotconv()

    # Return as Spec1D object, 1 order
    synspec = Spec1D(flux/cont,wave=wave,lsfpars=np.array(0.0))
    synspec.cont = cont
    
    return synspec


def prepare_synthspec(spec,lsf):
    """ Prepare a synthetic spectrum to be compared to an observed spectrum."""
    # convolve with LSF

    ## Get full wavelength range and total wavelength coverage in the orders
    #owr = dln.minmax(lsf.wave)
    #owavefull = dln.valrange(lsf.wave)
    #owavechunks = 0.0
    #odw = np.zeros(lsf.norder,np.float64)
    #specwave = np.atleast_2d(lsf.wave.copy())
    #for o in range(spec.norder):
    #    owavechunks += dln.valrange(specwave[:,o])
    #    odw[o] = np.median(dln.slope(specwave[:,o]))

    # Initialize the output spectrum
    npix,norder = lsf.wave.shape
    pspec = Spec1D(np.zeros((npix,norder),np.float32),wave=lsf.wave,lsfpars=lsf.pars,lsftype=lsf.lsftype,lsfxtype=lsf.xtype)
    pspec.cont = np.zeros((npix,norder),np.float32)
        
    # Loop over orders
    for o in range(lsf.norder):
        wobs = lsf.wave[:,o]
        dw = np.median(dln.slope(wobs))
        wv1,ind1 = dln.closest(spec.wave,np.min(wobs)-2*np.abs(dw))
        wv2,ind2 = dln.closest(spec.wave,np.max(wobs)+2*np.abs(dw))
        modelflux = spec.flux[ind1:ind2+1]
        modelwave = spec.wave[ind1:ind2+1]
        modelcont = spec.cont[ind1:ind2+1]

        # Rebin, if necessary
        #  get LSF FWHM (A) for a handful of positions across the spectrum
        xp = np.arange(npix//20)*20
        fwhm = lsf.fwhm(wobs[xp],xtype='Wave',order=o)
        # FWHM is in units of lsf.xtype, convert to wavelength/angstroms, if necessary
        if lsf.xtype.lower().find('pix')>-1:
            fwhm *= np.abs(dw)
        #  convert FWHM (A) in number of model pixels at those positions
        dwmod = dln.slope(modelwave)
        dwmod = np.hstack((dwmod,dwmod[-1]))
        xpmod = interp1d(modelwave,np.arange(len(modelwave)),kind='cubic',bounds_error=False,
                         fill_value=(np.nan,np.nan),assume_sorted=False)(wobs[xp])
        xpmod = np.round(xpmod).astype(int)
        fwhmpix = np.abs(fwhm/dwmod[xpmod])
        # need at least ~4 pixels per LSF FWHM across the spectrum
        #  using 3 affects the final profile shape
        nbin = np.round(np.min(fwhmpix)//4).astype(int)
        if nbin>1:
            npix2 = np.round(len(spec.flux) // nbin).astype(int)
            modelflux = dln.rebin(modelflux[0:npix2*nbin],npix2)
            modelwave = dln.rebin(modelwave[0:npix2*nbin],npix2)
            modelcont = dln.rebin(modelcont[0:npix2*nbin],npix2)            
        
        # Convolve
        lsf2d = lsf.anyarray(modelwave,xtype='Wave',order=o,original=False)
        cflux = utils.convolve_sparse(modelflux,lsf2d)
        # Interpolate onto final wavelength array
        flux = synple.interp_spl(wobs, modelwave, cflux)
        cont = synple.interp_spl(wobs, modelwave, modelcont)
        pspec.flux[:,o] = flux
        pspec.cont[:,o] = cont        

    return pspec


def mkbounds(params):
    """ Make lower and upper boundaries for parameters """
    params = np.char.array(params).upper()
    n = len(params)
    lbounds = np.zeros(n,np.float64)
    ubounds = np.zeros(n,np.float64)    
    # Teff
    g, = np.where(params=='TEFF')
    if len(g)>0:
        lbounds[g[0]] = 3500
        ubounds[g[0]] = 60000
    # logg
    g, = np.where(params=='LOGG')
    if len(g)>0:
        lbounds[g[0]] = 0
        ubounds[g[0]] = 5    
    # fe_h
    g, = np.where(params=='FE_H')
    if len(g)>0:
        lbounds[g[0]] = -3
        ubounds[g[0]] = 1       
    # Vmicro
    g, = np.where(params=='LOGG')
    if len(g)>0:
        lbounds[g[0]] = 0
        ubounds[g[0]] = 5        
    # Vsini/vrot
    g, = np.where(params=='VROT')
    if len(g)>0:
        lbounds[g[0]] = 0
        ubounds[g[0]] = 500        
    # RV
    g, = np.where(params=='RV')
    if len(g)>0:
        lbounds[g[0]] = -1500
        ubounds[g[0]] = 1500    
    # abundances
    g, = np.where( (params.find('_H') != -1) & (params != 'FE_H') )
    if len(g)>0:
        lbounds[g] = -3
        ubounds[g] = 5       

    bounds = (lbounds,ubounds)
    return bounds


def fit(spec,allparams,fitparams):
    """ Fit a spectrum and determine the abundances."""

    #allparams = {'teff':3500.0,'logg':2.5,'fe_h':-1.2,'rv':0.0,'ca_h':-0.5}
    #fitparams = ['teff','logg','fe_h','rv']

    # Capitalize the inputs
    # Make key names all CAPS
    allparams = dict((key.upper(), value) for (key, value) in allparams.items())
    fitparams = [v.upper() for v in fitparams]

    # Normalize the spectrum
    if spec.normalize==False:
        spec.normalize()
    
    # Use doppler to get initial guess of stellar parameters and RV
    print('Running Doppler')
    dopout, dopfmodel, dopspecm = doppler.fit(spec)
    print('Teff = %f' % dopout['teff'][0])
    print('logg = %f' % dopout['logg'][0])
    print('[Fe/H] = %f' % dopout['feh'][0])
    print('Vrel = %f' % dopout['vrel'][0])
    
    # Initialize the fitter
    allparams['TEFF'] = dopout['teff'][0]
    allparams['LOGG'] = dopout['logg'][0]
    allparams['FE_H'] = dopout['feh'][0]
    allparams['RV'] = dopout['vrel'][0]
    spfitter = SpecFitter(spec,allparams)
    spfitter.fitparams = fitparams
    pinit = [allparams[k] for k in fitparams]
    bounds = mkbounds(fitparams)
    
    import pdb; pdb.set_trace()

    pars, cov = curve_fit(spfitter.model,spec.wave.flatten(),spec.flux.flatten(),
                          sigma=spec.err.flatten(),p0=pinit,bounds=bounds)

    import pdb; pdb.set_trace()
    
    
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
