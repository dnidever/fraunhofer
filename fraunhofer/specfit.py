#!/usr/bin/env python

"""SPECFIT.PY - Generic stellar abundance determination software

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200711'  # yyyymmdd

import os
import shutil
import contextlib, io, sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
from dlnpyutils.minpack import curve_fit
from dlnpyutils.least_squares import least_squares
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln, bindata, astro
import doppler
from doppler.spec1d import Spec1D
from doppler import (cannon,utils,reader)
import copy
import logging
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import tempfile
from . import models
from synple import synple

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


cspeed = 2.99792458e5  # speed of light in km/s

class SpecFitter:
    def __init__ (self,spec,allparams,fitparams=None,verbose=False):
        # Parameters
        self.allparams = allparams
        if fitparams is not None:
            self.fitparams = fitparams
        else:
            self.fitparams = list(allparams.keys())  # by default fit all parameters            
        self.nsynfev = 0  # number of synthetic spectra made
        self.njac = 0     # number of times jacobian called
        # Save spectrum information    
        self.spec = spec.copy()
        self.flux = spec.flux.flatten()
        self.err = spec.err.flatten()
        self.wave = spec.wave.flatten()
        self.lsf = spec.lsf.copy()
        self.lsf.wavevac = spec.wavevac  # need this later for synspec prep
        self.wavevac = spec.wavevac
        self.verbose = verbose
        # Convert vacuum to air wavelengths
        #  synspec uses air wavelengths
        wave = spec.wave.copy()
        if spec.wavevac is True:
            wave = astro.vactoair(wave.flatten()).reshape(spec.wave.shape)
        # Figure out the wavelength parameters
        npix,norder = spec.flux.shape
        xp = np.arange(npix//20)*20
        wr = np.zeros((spec.lsf.norder,2),np.float64)
        dw = np.zeros(spec.lsf.norder,np.float64)
        mindw = np.zeros(norder,np.float64)
        for o in range(spec.norder):
            dw[o] = np.median(dln.slope(wave[:,o]))
            wr[o,0] = np.min(wave[:,o])
            wr[o,1] = np.max(wave[:,o])            
            fwhm = spec.lsf.fwhm(spec.wave[xp,o],xtype='Wave',order=o)
            # FWHM is in units of lsf.xtype, convert to wavelength/angstroms, if necessary
            if spec.lsf.xtype.lower().find('pix')>-1:
                fwhm *= np.abs(dw[o])

            # need at least ~4 pixels per LSF FWHM across the spectrum
            #  using 3 affects the final profile shape
            mindw[o] = np.min(fwhm/4)
        self._dwair = np.min(mindw)    # IN AIR WAVELENGTHS!!
        self._w0air = np.min(wave)
        self._w1air = np.max(wave)
        # parameters to save
        self._all_pars = []
        self._all_model = []
        self._all_chisq = []
        self._jac_array = None

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
        inputs['DW'] = self._dwair          # add in wavelength parameters
        inputs['W0'] = self._w0air
        inputs['W1'] = self._w1air
        return inputs

    def chisq(self,model):
        return np.sqrt( np.sum( (self.flux-model)**2/self.err**2 )/len(self.flux) )
        
    def model(self, xx, *args):
        """ Return a model spectrum flux with the given input arguments."""
        # This corrects for air/vacuum wavelength differences
        if self.verbose:
            print(args)
        # The arguments correspond to the fitting parameters
        inputs = self.mkinputs(args)
        if self.verbose:
            print(inputs)
        # Create the synthetic spectrum
        synspec = model_spectrum(inputs,verbose=self.verbose)   # always returns air wavelengths
        self.nsynfev += 1
        # Convolve with the LSF and do air/vacuum wave conversion
        pspec = prepare_synthspec(synspec,self.lsf)
        # Save models/pars/chisq
        self._all_pars.append(list(args).copy())
        self._all_model.append(pspec.flux.flatten().copy())
        self._all_chisq.append(self.chisq(pspec.flux.flatten()))
        # Return flattened spectrum        
        return pspec.flux.flatten()

    def getstep(self,name,val,relstep=0.02):
        """ Calculate step for a parameter."""
        # It mainly deals with edge cases
        if val != 0.0:
            step = relstep*val
        else:
            if name=='RV':
                step = 1.0
            elif name=='VROT':
                step = 0.5
            elif name=='VMICRO':
                step = 0.5
            elif name.endswith('_H'):
                step = 0.02
            else:
                step = 0.02
        return step
                
    def jac(self,x,*args):
        """ Compute the Jacobian matrix (an m-by-n matrix, where element (i, j)
        is the partial derivative of f[i] with respect to x[j]). """

        if hasattr(self,'logger') is False:
            logger = dln.basiclogger()
        else:
            logger = self.logger
            
        logger.info(args)
        
        if self.verbose:
            logger.info(' ')
            logger.info('##### Calculating Jacobian Matrix #####')
            logger.info(' ')
            
        # A new synthetic spectrum does not need to be generated RV, vmicro or vsini.
        # Some time can be saved by not remaking those.
        # Use a one-sided derivative.

        # Boundaries
        lbounds,ubounds = mkbounds(self.fitparams)
        
        relstep = 0.02
        npix = len(x)
        npar = len(args)

        # Get INPUTS dictionary and make keys all CAPS
        inputs = self.mkinputs(args)
        inputs = dict((key.upper(), value) for (key, value) in inputs.items())

        # Some important parameters
        w0 = inputs['W0']
        w1 = inputs['W1']
        dw = inputs['DW']
        rv = inputs.get('RV')
        vrot = inputs.get('VROT')
        vmicro = inputs.get('VMICRO') 
        
        # Create synthetic spectrum at current values
        #  set vrot=vmicro=rv=0, will modify later if necessary
        if self.verbose:
            logger.info('--- Current values ---')
            logger.info(args)
        tinputs = inputs.copy()
        tinputs['VMICRO'] = 0
        tinputs['VROT'] = 0
        tinputs['RV'] = 0
        origspec = model_spectrum(tinputs,keepextend=True)  # always are wavelengths
        self.nsynfev += 1
        # Smooth and shift
        smorigspec = smoothshift_spectrum(origspec,vrot=vrot,vmicro=vmicro,rv=rv)
        # Trim to final wavelengths
        smorigspec = trim_spectrum(smorigspec,w0,w1)
        # Convolve with the LSF and do air/vacuum wave conversion
        pspec = prepare_synthspec(smorigspec,self.lsf)
        # Flatten the spectrum
        f0 = pspec.flux.flatten()
        # Save models/pars/chisq
        self._all_pars.append(list(args).copy())
        self._all_model.append(f0.copy())
        self._all_chisq.append(self.chisq(f0))
        chisq = np.sqrt( np.sum( (self.flux-f0)**2/self.err**2 )/len(self.flux) )
        self
        if self.verbose:
            logger.info('chisq = '+str(chisq))
        
        # MASK PIXELS!?
        
        # Initialize jacobian matrix
        jac = np.zeros((npix,npar),np.float64)
        
        # Loop over parameters
        for i in range(npar):
            pars = np.array(copy.deepcopy(args))
            step = self.getstep(self.fitparams[i],pars[i],relstep)
            # Check boundaries, if above upper boundary
            #   go the opposite way
            if pars[i]>ubounds[i]:
                step *= -1
            pars[i] += step
            tinputs = self.mkinputs(pars)

            if self.verbose:
                logger.info(' ')
                logger.info('--- '+str(i+1)+' '+self.fitparams[i]+' '+str(pars[i])+' ---')
                logger.info(pars)
            # VROT/VMICRO/RV, just shift/smooth original spectrum
            if self.fitparams[i]=='VROT' or self.fitparams[i]=='VMICRO' or self.fitparams[i]=='RV':
                tvrot = tinputs.get('VROT')
                tvmicro = tinputs.get('VMICRO')
                trv = tinputs.get('RV')
                #import pdb; pdb.set_trace()                
                # Smooth and shift
                synspec = smoothshift_spectrum(origspec,vrot=tvrot,vmicro=tvmicro,rv=trv)
                # Trim to final wavelengths
                synspec = trim_spectrum(synspec,w0,w1)
            else:
                synspec = model_spectrum(tinputs)  # always returns air wavelengths
                self.nsynfev += 1
                
            # Convert to vacuum wavelengths if necessary
            if self.wavevac:
                synspec.wave = astro.airtovac(synspec.wave)
                synspec.wavevac = True
            # Convolve with the LSF and do air/vacuum wave conversion
            pspec = prepare_synthspec(synspec,self.lsf)
            # Flatten the spectrum
            f1 = pspec.flux.flatten()

            # Save models/pars/chisq
            self._all_pars.append(list(pars).copy())
            self._all_model.append(f1.copy())
            self._all_chisq.append(self.chisq(f1))
            
            if np.sum(~np.isfinite(f1))>0:
                print('some nans/infs')
                import pdb; pdb.set_trace()

            jac[:,i] = (f1-f0)/step

        if np.sum(~np.isfinite(jac))>0:
            print('some nans/infs')
            import pdb; pdb.set_trace()

        self._jac_array = jac.copy()   # keep a copy
            
        self.njac += 1
            
        return jac

    
def trim_spectrum(spec,w0,w1):
    """ Trim a synthetic spectrum to [w0,w1]."""
    # This assumes that the spectrum has a single order
    wv1, ind1 = dln.closest(spec.wave,w0)
    wv2, ind2 = dln.closest(spec.wave,w1)
    # Nothing to do
    if ind1==0 and ind2==(spec.npix-1):
        return spec
    outspec = spec.copy()
    outspec.flux = outspec.flux[ind1:ind2+1]
    outspec.wave = outspec.wave[ind1:ind2+1]
    if outspec.err is not None:
        outspec.err = outspec.err[ind1:ind2+1]
    if outspec.mask is not None:
        outspec.mask = outspec.mask[ind1:ind2+1]
    if hasattr(outspec,'cont'):
        if outspec.cont is not None:
            outspec.cont = outspec.cont[ind1:ind2+1]        
    outspec.npix = len(outspec.flux)
    return outspec


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
    atmostype, teff, logg, vmicro2, mabu, nd, atmos = synple.read_model(modelfile,verbose=verbose)
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

    # Deal with alpha abundances
    #  only add the individual alpha abundance if it's not already there
    #  sometimes we might fit a single alpha element but want to use
    #  ALPHA_H to set the rest of them
    if inputs.get('ALPHA_H') is not None:
        alpha = inputs['ALPHA_H']
        elem = ['O','MG','SI','S','CA','TI']
        for k in range(len(elem)):
            if inputs.get(elem[k]+'_H') is None:
                inputs[elem[k]+'_H'] = alpha
    
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
    # Wavelengths are all AIR!!
    
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
    os.close(tid)  # close the open file
    # Limit values
    #  of course the logg/feh ranges vary with Teff
    mteff = dln.limit(teff,3500.0,60000.0)
    mlogg = dln.limit(logg,0.0,5.0)
    mmetal = dln.limit(metal,-2.5,0.5)
    model, header, tail = models.mkmodel(mteff,mlogg,mmetal,modelfile)
    inputs['modelfile'] = modelfile
    if os.path.exists(modelfile) is False or os.stat(modelfile).st_size==0:
        print('model atmosphere file does NOT exist')
        import pdb; pdb.set_trace()
    
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
    
    wave,flux,cont = synple.syn(modelfile,(w0,w1),dw,vmicro=vmicro,vrot=vrot,
                                abu=list(abu),verbose=verbose)

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
        flux = utils.broaden(spec.wave,spec.flux,vgauss=vmicro,vsini=vrot)
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
                if np.sum(np.isfinite(spec.flux[gd]))>0:
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
    
    The returned spectrum always uses AIR wavelengths!!!

    """
    
    # Make key names all CAPS
    inputs = dict((key.upper(), value) for (key, value) in inputs.items())

    # Extend on the ends for RV/convolution purposes
    w0 = inputs['W0']
    w1 = inputs['W1']
    dw = inputs['DW']
    rv = inputs.get('RV')
    vrot = inputs.get('VROT')
    vmicro = inputs.get('VMICRO')
    inputsext = inputs.copy()
    if rv is not None or vrot is not None or vmicro is not None:
        numext = int(np.ceil(w1*(1.0+1500/cspeed)-w1))
        inputsext['W0'] = w0-numext*dw
        inputsext['W1'] = w1+numext*dw
        if verbose:
            print('Extending wavelength by '+str(numext)+' pixels on each end')
        
    # Create the synthetic spectrum
    #  set vrot=vmicro=0, will convolve later if necessary
    inputsext['VMICRO'] = 0
    inputsext['VROT'] = 0
    wave1,flux1,cont1 = synple_wrapper(inputsext,verbose=verbose)

    # Get final wavelength array
    wv1, ind1 = dln.closest(wave1,w0)
    wv2, ind2 = dln.closest(wave1,w1)
    synspec = Spec1D(flux1/cont1,wave=wave1,lsfpars=np.array(0.0))
    synspec.cont = cont1
    synspec.wavevac = False
    # Smooth and shift
    if rv is not None or vrot is not None or vmicro is not None:
        synspec = smoothshift_spectrum(synspec,vrot=vrot,vmicro=vmicro,rv=rv)
    # Trim to final wavelengths
    if keepextend is False:
        synspec = trim_spectrum(synspec,w0,w1)

    return synspec


def prepare_synthspec(synspec,lsf):
    """ Prepare a synthetic spectrum to be compared to an observed spectrum."""
    # Convolve with LSF and do air<->vacuum wavelength conversion
    
    # Convert wavelength from air->vacuum or vice versa
    if synspec.wavevac != lsf.wavevac:
        # Air -> Vacuum
        if synspec.wavevac is False:
            synspec.wave = astro.airtovac(synspec.wave)
            synspec.wavevac = True
        # Vacuum -> Air
        else:
            synspec.dispersion = astro.vactoair(synspec.wave)
            synspec.wavevac = False
        
    # Initialize the output spectrum
    npix,norder = lsf.wave.shape
    pspec = Spec1D(np.zeros((npix,norder),np.float32),wave=lsf.wave,lsfpars=lsf.pars,lsftype=lsf.lsftype,lsfxtype=lsf.xtype)
    pspec.cont = np.zeros((npix,norder),np.float32)
        
    # Loop over orders
    for o in range(lsf.norder):
        wobs = lsf.wave[:,o]
        dw = np.median(dln.slope(wobs))
        wv1,ind1 = dln.closest(synspec.wave,np.min(wobs)-2*np.abs(dw))
        wv2,ind2 = dln.closest(synspec.wave,np.max(wobs)+2*np.abs(dw))
        modelflux = synspec.flux[ind1:ind2+1]
        modelwave = synspec.wave[ind1:ind2+1]
        modelcont = synspec.cont[ind1:ind2+1]

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
            npix2 = np.round(len(synspec.flux) // nbin).astype(int)
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
    g, = np.where(params=='VMICRO')
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


def mkdxlim(fitparams):
    """ Make array of parameter changes at which curve_fit should finish."""
    npar = len(fitparams)
    dx_lim = np.zeros(npar,float)
    for k in range(npar):
        if fitparams[k]=='TEFF':
            dx_lim[k] = 1.0
        elif fitparams[k]=='LOGG':
            dx_lim[k] = 0.005
        elif fitparams[k]=='VMICRO':
            dx_lim[k] = 0.1
        elif fitparams[k]=='VROT':
            dx_lim[k] = 0.1
        elif fitparams[k]=='RV':
            dx_lim[k] = 0.01
        elif fitparams[k].endswith('_H'):
            dx_lim[k] = 0.005
        else:
            dx_lim[k] = 0.01
    return dx_lim

    
def initpars(allparams,fitparams):
    """ Make initial set of parameters given ALLPARAMS and
        FITPARAMS."""

    allparams = dict((key.upper(), value) for (key, value) in allparams.items()) # all CAPS
    fitparams = [v.upper() for v in fitparams]  # all CAPS
    
    npars = len(fitparams)
    pinit = np.zeros(npars,np.float64)
    # Loop over parameters
    for k in range(npars):
        ind, = np.where(np.char.array(list(allparams.keys()))==fitparams[k])
        # This parameter is in ALLPARAMS
        if len(ind)>0:
            pinit[k] = allparams[fitparams[k]]
        # Not in ALLPARAMS
        else:
            if fitparams[k]=='RV':
                pinit[k] = 0.0
            elif fitparams[k]=='VMICRO':
                pinit[k] = 2.0
            elif fitparams[k]=='VROT':
                pinit[k] = 0.0
            elif fitparams[k]=='TEFF':
                pinit[k] = 5000.0
            elif fitparams[k]=='LOGG':
                pinit[k] = 3.0
            elif fitparams[k].endswith('_H'):
                # Abundances, use FE_H if possible
                if 'FE_H' in allparams.keys():
                    pinit[k] = allparams['FE_H']
                else:
                    pinit[k] = 0.0
            else:
                pinit[k] = 0.0

    return pinit



def specfigure(figfile,spec,fmodel,out,original=None,verbose=True,figsize=10):
    """ Make diagnostic figure."""
    #import matplotlib
    matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    if os.path.exists(figfile): os.remove(figfile)
    norder = spec.norder
    nlegcol = 2
    if original is not None: nlegcol=3
    # Single-order plot
    if norder==1:
        fig,ax = plt.subplots()
        fig.set_figheight(figsize*0.5)
        fig.set_figwidth(figsize)
        if original is not None:
            plt.plot(original.wave,original.flux,color='green',label='Original',linewidth=1)
        plt.plot(spec.wave,spec.flux,'b',label='Masked Data',linewidth=1)
        plt.plot(fmodel.wave,fmodel.flux,'r',label='Model',linewidth=1,alpha=0.8)
        leg = ax.legend(loc='upper left', frameon=True, framealpha=0.8, ncol=nlegcol)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        xr = dln.minmax(spec.wave)
        yr = [np.min([spec.flux,fmodel.flux]), np.max([spec.flux,fmodel.flux])]
        if original is not None:
            yr = [np.min([original.flux,spec.flux,fmodel.flux]), np.max([spec.flux,fmodel.flux])]            
        yr = [yr[0]-dln.valrange(yr)*0.15,yr[1]+dln.valrange(yr)*0.005]
        yr = [np.max([yr[0],-0.2]), np.min([yr[1],2.0])]
        plt.xlim(xr)
        plt.ylim(yr)
        snr = np.nanmedian(spec.flux/spec.err)
        plt.title(spec.filename)
        #ax.annotate(r'S/N=%5.1f   Teff=%5.1f$\pm$%5.1f  logg=%5.2f$\pm$%5.2f  [Fe/H]=%5.2f$\pm$%5.2f   Vrel=%5.2f$\pm$%5.2f   chisq=%5.2f' %
        #            (snr, out['TEFF'], out['tefferr'], out['LOGG'], out['loggerr'], out['FE_H'], out['feherr'], out['RV'], out['vrelerr'], out['chisq']),
        #            xy=(np.mean(xr), yr[0]+dln.valrange(yr)*0.05),ha='center')
    # Multi-order plot
    else:
        fig,ax = plt.subplots(norder)
        fig.set_figheight(figsize)
        fig.set_figwidth(figsize)
        for i in range(norder):
            if original is not None:
                ax[i].plot(original.wave[:,i],original.flux[:,i],color='green',label='Original',linewidth=1)            
            ax[i].plot(spec.wave[:,i],spec.flux[:,i],'b',label='Masked Data',linewidth=1)
            ax[i].plot(fmodel.wave[:,i],fmodel.flux[:,i],'r',label='Model',linewidth=1,alpha=0.8)
            if i==0:
                leg = ax[i].legend(loc='upper left', frameon=True, framealpha=0.8, ncol=nlegcol)
            ax[i].set_xlabel('Wavelength (Angstroms)')
            ax[i].set_ylabel('Normalized Flux')
            xr = dln.minmax(spec.wave[:,i])
            yr = [np.min([spec.flux[:,i],fmodel.flux[:,i]]), np.max([spec.flux[:,i],fmodel.flux[:,i]])]
            if original is not None:
                yr = [np.min([original.flux[:,i],spec.flux[:,i],fmodel.flux[:,i]]), np.max([spec.flux[:,i],fmodel.flux[:,i]])]
            yr = [yr[0]-dln.valrange(yr)*0.05,yr[1]+dln.valrange(yr)*0.05]
            if i==0:
                yr = [yr[0]-dln.valrange(yr)*0.15,yr[1]+dln.valrange(yr)*0.05]            
            yr = [np.max([yr[0],-0.2]), np.min([yr[1],2.0])]
            ax[i].set_xlim(xr)
            ax[i].set_ylim(yr)
            # legend
            if i==0:
                snr = np.nanmedian(spec.flux/spec.err)
                ax[i].set_title(spec.filename)
                #ax[i].annotate(r'S/N=%5.1f   Teff=%5.1f$\pm$%5.1f  logg=%5.2f$\pm$%5.2f  [Fe/H]=%5.2f$\pm$%5.2f   Vrel=%5.2f$\pm$%5.2f   chisq=%5.2f' %
                #               (snr,out['teff'],out['tefferr'],out['logg'],out['loggerr'],out['feh'],out['feherr'],out['vrel'],out['vrelerr'],out['chisq']),
                #               xy=(np.mean(xr), yr[0]+dln.valrange(yr)*0.05),ha='center')
    plt.savefig(figfile,bbox_inches='tight')
    plt.close(fig)
    if verbose is True: print('Figure saved to '+figfile)


def dopvrot_lsq(spec,models=None,initpar=None,verbose=False,logger=None):
    """
    Least Squares fitting with forward modeling of the spectrum.
    
    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    models : list of Cannon models, optional
         A list of Cannon models to use.  The default is to load all of the Cannon
         models in the data/ directory and use those.
    initpar : numpy array, optional
         Initial estimate for [teff, logg, feh, RV, vsini], optional.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Cannon model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = fit_lsq(spec)

    """
    
    if logger is None:
        logger = dln.basiclogger()

    # Load and prepare the Cannon models
    #-------------------------------------------
    if models is None:
        models = cannon.models.copy()
        models.prepare(spec)
    
    # Get initial estimates
    if initpar is None:
        initpar = np.array([6000.0, 2.5, -0.5, 0.0, 0.0])
    initpar = np.array(initpar).flatten()
    
    # Calculate the bounds
    lbounds = np.zeros(5,float)+1e5
    ubounds = np.zeros(5,float)-1e5
    for p in models:
        lbounds[0:3] = np.minimum(lbounds[0:3],np.min(p.ranges,axis=1))
        ubounds[0:3] = np.maximum(ubounds[0:3],np.max(p.ranges,axis=1))
    lbounds[3] = -1000
    ubounds[3] = 1000
    lbounds[4] = 0.0
    ubounds[4] = 500.0
    bounds = (lbounds, ubounds)
    
    # function to use with curve_fit
    def spec_interp_vsini(x,teff,logg,feh,rv,vsini):
        """ This returns the interpolated model for a given spectrum."""
        # The "models" and "spec" must already exist outside of this function
        m = models(teff=teff,logg=logg,feh=feh,rv=rv)
        if m is None:      # there was a problem
            return np.zeros(spec.flux.shape,float).flatten()+1e30
        # Broaden to vsini
        if spec.norder>1:
            smflux = spec.flux*0
            for k in range(spec.norder):
                smflux[:,k] = utils.broaden(m.wave[:,k],m.flux[:,k],vsini=vsini)            
        else:
            smflux = utils.broaden(m.wave.flatten(),m.flux.flatten(),vsini=vsini)
        return smflux.flatten()
    
    # Use curve_fit
    lspars, lscov = curve_fit(spec_interp_vsini, spec.wave.flatten(), spec.flux.flatten(), sigma=spec.err.flatten(),
                              p0=initpar, bounds=bounds)
    # If it hits a boundary then the solution won't change much compared to initpar
    # setting absolute_sigma=True gives crazy low lsperror values
    lsperror = np.sqrt(np.diag(lscov))
    
    if verbose is True:
        logger.info('Least Squares RV and stellar parameters:')
        for k,n in enumerate(['Teff','logg','[Fe/H]','RV','Vsini']):
            logger.info('%s = %f' % (n,lspars[k]))
    lsmodel = spec_interp_vsini(spec.wave,teff=lspars[0],logg=lspars[1],feh=lspars[2],rv=lspars[3],vsini=lspars[4])
    lschisq = np.sqrt(np.sum(((spec.flux.flatten()-lsmodel)/spec.err.flatten())**2)/len(lsmodel))
    if verbose is True: logger.info('chisq = %5.2f' % lschisq)

    # Put it into the output structure
    npar = len(lspars)
    dtype = np.dtype([('pars',float,npar),('parerr',float,npar),('parcov',float,(npar,npar)),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['pars'] = lspars
    out['parerr'] = lsperror
    out['parcov'] = lscov
    out['chisq'] = lschisq
    
    return out, lsmodel


def fit_lsq(spec,allparams,fitparams=None,verbose=0,logger=None):
    """
    Fit a spectrum with a synspec synthetic spectrum and determine stellar parameters and
    abundances using least-squares.

    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    allparams : dict
         Dictionary of initial values to use or parameters/elements to hold fixed.
    fitparams : list, optional
         List of parameter names to fit (e.g., TEFF, LOGG, FE_H, RV).  By default all values
         in ALLPARAMS are fit.
    verbose : int, optional
         Verbosity level (0, 1, or 2).  The default is 0 and verbose=2 is for debugging.
    logger : logging object, optional
         Logging object.

    Returns
    -------
    out : numpy structured array
        Catalog of best-fit values.
    model : numpy array
        The best-fit synthetic stellar spectrum.

    Example
    -------

    .. code-block:: python

         spec = doppler.read(file)
         allparams = {'teff':5500,'logg':3.0,'fe_h':-1.0,'rv':0.0,'ca_h':-1.0}
         fitparams = ['teff','logg','fe_h','rv','ca_h']
         out,model = specfit.fit_lsq(spec,allparams,fitparams=fitparams)

    """


    t0 = time.time()
    
    if logger is None:
        logger = dln.basiclogger()
    
    # Normalize the spectrum
    if spec.normalized==False:
        spec.normalize()
    
    # Capitalize the inputs
    # Make key names all CAPS
    allparams = dict((key.upper(), value) for (key, value) in allparams.items())

    # Fitting parameters
    if fitparams is None:
        fitparams = list(allparams.keys())
    fitparams = [v.upper() for v in fitparams]  # all CAPS
    npar = len(fitparams)
    
    # Initialize the fitter
    spfitter = SpecFitter(spec,allparams,fitparams=fitparams,verbose=(verbose>=2))
    spfitter.logger = logger
    pinit = initpars(allparams,fitparams)
    bounds = mkbounds(fitparams)

    if verbose>0:
        logger.info('Fitting: '+', '.join(fitparams))
        
    # Fit the spectrum using curve_fit
    dx_lim = mkdxlim(fitparams)
    pars, cov = curve_fit(spfitter.model,spfitter.wave,spfitter.flux,dx_lim=dx_lim,
                          sigma=spfitter.err,p0=pinit,bounds=bounds,jac=spfitter.jac)
    error = np.sqrt(np.diag(cov))

    if verbose>0:
        logger.info('Best values:')
        for k in range(npar):
            logger.info('%s = %f' % (fitparams[k],pars[k]))
    model = spfitter.model(spfitter.wave,*pars)
    chisq = np.sqrt(np.sum(((spfitter.flux-model)/spfitter.err)**2)/len(model))
    if verbose>0:
        logger.info('chisq = %5.2f' % chisq)
        logger.info('nfev = %i' % spfitter.nsynfev)
        logger.info('dt = %f sec.' % (time.time()-t0))
        
    # Put it into the output structure
    dtyp = []
    for f in fitparams:
        dtyp += [(f,float),(f+'_ERR',float)]
    dtyp += [('pars',float,npar),('parerr',float,npar),('parcov',float,(npar,npar)),('chisq',float),('nsynfev',int)]
    dtype = np.dtype(dtyp)
    out = np.zeros(1,dtype=dtype)
    for k,f in enumerate(fitparams):
        out[f] = pars[k]
        out[f+'_ERR'] = error[k]
    out['pars'] = pars
    out['parerr'] = error
    out['parcov'] = cov
    out['chisq'] = chisq
    out['nsynfev'] = spfitter.nsynfev

    # Reshape final model spectrum
    model = model.reshape(spec.flux.shape)
    
    return out, model



def fit(spec,allparams=None,fitparams=None,elem=None,figfile=None,fitvsini=False,fitvmicro=False,
        verbose=None,logger=None):
    """
    Fit a spectrum with a synspec synthetic spectrum and determine stellar parameters and
    abundances using a multi-step iterative method.

    Step 1: Fit Teff/logg/[Fe/H]/RV using Doppler
    Step 2: Fit Teff/logg/[Fe/H]/RV + vsini with Doppler model
    Step 3: Fit stellar parameters (Teff/logg/[Fe/H]/[alpha/H]), RV and broadening (Vrot/Vmicro)
    Step 4: Fit each element one at a time holding everything else fixed.
    Step 5: Fit everything simultaneously

    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    allparams : dict, optional
         Dictionary of initial values to use or parameters/elements to hold fixed.
    elem : list, optional
         List of elements to fit.  The default is:
         elem = ['C','N','O','NA','MG','AL','SI','K','CA','TI','V','CR','MN','CO','NI','CU','CE','ND']
         Input an empty list [] to fit no elements.
    figfile : string, optional
         The filename for a diagnostic plot showing the observed spectrum and model spectrum.
    fitvsini : bool, optional
         Fit rotational velocity (vsini).  By default, Vsini will be fit initially with a Doppler
           model, but only included in the final fit if it improved chisq.
    fitvmicro : bool, optional
         Fit Vmicro.  Default is False.  By default, Vmicro is set (if not included in ALLPARAMS)
         logg>=3.8:  vmicro = 2.0
         logg<3.8:   vmicro = 10^(0.226−0.0228*logg+0.0297*(logg)^2−0.0113*(logg)^3 )
    verbose : int, optional
         Verbosity level (0, 1, or 2).  The default is 0 and verbose=2 is for debugging.
    logger : logging object, optional
         Logging object.

    Returns
    -------
    out : numpy structured array
        Catalog of best-fit values.
    model : numpy array
        The best-fit synthetic stellar spectrum.

    Example
    -------

    .. code-block:: python

         spec = doppler.read(file)
         out,model = specfit.fit(spec)

    """

    t0 = time.time()
    
    if logger is None:
        logger = dln.basiclogger()
        logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.handlers[0].setStream(sys.stdout)  
    
    # Default set of elements
    if elem is None:
        elem = ['C','N','O','NA','MG','AL','SI','K','CA','TI','V','CR','MN','CO','NI','CU','CE','ND']
    
    # Normalize the spectrum
    if spec.normalized==False:
        spec.normalize()

    # Print out inputs
    if verbose>0:
        logger.info('Inputs:')
        if allparams is not None:
            logger.info('ALLPARAMS:')
            for k,n in enumerate(allparams.keys()):
                logger.info('%s = %f' % (n,allparams[n]))
        else:
            logger.info('ALLPARAMS: None')
        if fitvmicro:
            logger.info('Fitting VMICRO')
        if fitvsini:
            logger.info('Fitting VSINI')            
        if len(elem)>0:
            logger.info('Elements to fit: '+', '.join(elem))
        else:
            logger.info('No elements to fit')
        logger.info(' ')
            

    # 1) Doppler (Teff, logg, feh, RV)
    #---------------------------------
    t1 = time.time()
    if verbose>0:
        logger.info('Step 1: Running Doppler')        
    # Use Doppler to get initial guess of stellar parameters and RV
    dopout, dopfmodel, dopspecm = doppler.fit(spec)
    if verbose>0:
        logger.info('Teff = %f' % dopout['teff'][0])
        logger.info('logg = %f' % dopout['logg'][0])
        logger.info('[Fe/H] = %f' % dopout['feh'][0])
        logger.info('Vrel = %f' % dopout['vrel'][0])
        logger.info('chisq = %f' % dopout['chisq'][0])
        logger.info('dt = %f sec.' % (time.time()-t1))
        # typically 5 sec
    

    # 2) Fit vsini as well with Doppler model
    #-----------------------------------------
    t2 = time.time()    
    if verbose>0:
        logger.info(' ')    
        logger.info('Step 2: Fitting vsini with Doppler model')
    initpar2 = [dopout['teff'][0], dopout['logg'][0], dopout['feh'][0], dopout['vrel'][0], 2.0]
    out2, model2 = dopvrot_lsq(spec,initpar=initpar2,verbose=verbose,logger=logger)
    if verbose>0:
        logger.info('Teff = %f' % out2['pars'][0][0])
        logger.info('logg = %f' % out2['pars'][0][1])
        logger.info('[Fe/H] = %f' % out2['pars'][0][2])
        logger.info('Vrel = %f' % out2['pars'][0][3])
        logger.info('Vsini = %f' % out2['pars'][0][4])
        logger.info('chisq = %f' % out2['chisq'][0])
        logger.info('dt = %f sec.' % (time.time()-t2))
        # typically 5 sec
    if out2['chisq'][0] > dopout['chisq'][0]:
        if verbose>0:
            logger.info('Doppler Vrot=0 chisq is better')
        out2['pars'][0] = [dopout['teff'][0],dopout['logg'][0],dopout['feh'][0],dopout['vrel'][0],0.0]

    # I'M NOT SURE THIS IS WORKING, VROT IS STAYING THE SAME

    
    # Initialize allparams
    if allparams is None:
        allparams = {}
    else:
        allparams = dict((key.upper(), value) for (key, value) in allparams.items())  # all CAPS
    # Using input values when possible, otherwise Doppler values
    for k,name in enumerate(['TEFF','LOGG','FE_H','RV','VROT']):
        if allparams.get(name) is None:
            allparams[name] = out2['pars'][0][k]
            
    #allparams['TEFF'] = out2['pars'][0][0]
    #allparams['LOGG'] = out2['pars'][0][1]
    #allparams['FE_H'] = out2['pars'][0][2]
    #allparams['RV'] = out2['pars'][0][3]
    #allparams['VROT'] = out2['pars'][0][4]


    # Get Vmicro using Teff/logg relation
    # APOGEE DR14 vmicro relation (Holtzman et al. 2018)
    # for stars with [M/H]>-1 and logg<3.8
    # vmicro = 10^(0.226−0.0228*logg+0.0297*(logg)^2−0.0113*(logg)^3 )
    # coef = [0.226,0.0228,0.0297,−0.0113]
    # only giants, was fit in dwarfs
    if allparams.get('VMICRO') is None:
        vmicro = 2.0  # default
        if allparams['LOGG']<3.8:
            vmcoef = [0.226,0.0228,0.0297,-0.0113]
            vmicro = 10**dln.poly(allparams['LOGG'],vmcoef[::-1])
        allparams['VMICRO'] = vmicro

    # for giants
    # vmacro = 10^(0.741−0.0998*logg−0.225[M/H])
    # maximum of 15 km/s

    
    # 3) Fit stellar parameters (Teff, logg, feh, alpha, RV, Vsini)
    #--------------------------------------------------------------
    t3 = time.time()    
    if verbose>0:
        logger.info(' ')
        logger.info('Step 3: Fitting stellar parameters, RV and broadening')
    allparams3 = allparams.copy()
    fitparams3 = ['TEFF','LOGG','FE_H','ALPHA_H','RV']    
    if allparams3['VROT']>0 or fitvsini is True:
        fitparams3.append('VROT')
    # Fit Vmicro as well if it's a dwarf
    if allparams3['LOGG']>3.8 or allparams3['TEFF']>8000 or fitvmicro is True:
        fitparams3.append('VMICRO')
    out3, model3 = fit_lsq(spec,allparams3,fitparams3,verbose=verbose,logger=logger)

    # Should we fit C_H and N_H as well??

    # typically 12-15 min.
    # now ~9 min.

    
    # Tweak the continuum
    if verbose is not None:
        logger.info('Tweaking continuum using best-fit synthetic model')
    tmodel = Spec1D(model3,wave=spec.wave.copy(),lsfpars=np.array(0.0))        
    spec = doppler.rv.tweakcontinuum(spec,tmodel)

    
    
    # 4) Fit each element separately
    #-------------------------------
    t4 = time.time()
    if verbose>0:
        logger.info(' ')
        logger.info('Step 4: Fitting each element separately')
    allparams4 = allparams3.copy()
    for k in range(len(fitparams3)):
        allparams4[fitparams3[k]] = out3['pars'][0][k]
    nelem = len(elem)
    if nelem>0:
        if verbose>0:
            logger.info('Elements: '+', '.join(elem))    
        elemcat = np.zeros(nelem,dtype=np.dtype([('name',np.str,10),('par',np.float64),('parerr',np.float64)]))
        elemcat['name'] = elem
        for k in range(nelem):
            t4b = time.time()
            allparselem = allparams4.copy()
            if elem[k] in ['O','MG','SI','S','CA','TI']:
                allparselem[elem[k]+'_H'] = allparams4['ALPHA_H']
            else:
                allparselem[elem[k]+'_H'] = allparams4['FE_H']
            fitparselem = [elem[k]+'_H']
            out4, model4 = fit_lsq(spec,allparselem,fitparselem,verbose=verbose,logger=logger)
            elemcat['par'][k] = out4['pars'][0]
            elemcat['parerr'][k] = out4['parerr'][0]
        if verbose>0:
            logger.info('dt = %f sec.' % (time.time()-t4))
    else:
        if verbose>0:
            logger.info('No elements to fit')
    # about 50 min.

              
    # 5) Fit all parameters simultaneously
    #---------------------------------------    
    # if NO elements to fit, then nothing to do
    if nelem>0:
        t5 = time.time()
        if verbose>0:
            logger.info('Step 5: Fit all parameters simultaneously')
        allparams5 = allparams4.copy()
        for k in range(nelem):
            allparams5[elem[k]+'_H'] = elemcat['par'][k]
        if allparams5.get('ALPHA_H') is not None:
            del allparams5['ALPHA_H']
        fitparams5 = ['TEFF','LOGG','FE_H','RV']
        if 'VROT' in fitparams3 or fitvsini is True:
            fitparams5.append('VROT')
        if 'VMICRO' in fitparams3 or fitvmicro is True:
            fitparams5.append('VMICRO')
        fitparams5 = fitparams5+list(np.char.array(elem)+'_H')
        out5, model5 = fit_lsq(spec,allparams5,fitparams5,verbose=verbose,logger=logger)
    else:
        out5 = out3
        model5 = model3
        fitparams5 = fitparams3


    # Make final structure and save the figure
    out = out5
    dtyp = []
    npar = len(fitparams5)
    for f in fitparams5:
        dtyp += [(f,float),(f+'_ERR',float)]
    dtyp += [('pars',float,npar),('parerr',float,npar),('parcov',float,(npar,npar)),('chisq',float)]
    dtype = np.dtype(dtyp)
    out = np.zeros(1,dtype=dtype)
    for k,f in enumerate(fitparams5):
        out[f] = out5['pars'][0][k]
        out[f+'_ERR'] = out5['parerr'][0][k]        
    out['pars'] = out5['pars'][0]
    out['parerr'] = out5['parerr'][0]
    out['parcov'] = out5['parcov'][0]
    out['chisq'] = out5['chisq'][0]
    # Final model
    model = Spec1D(model5,wave=spec.wave.copy(),lsfpars=np.array(0.0))
    model.lsf = spec.lsf.copy()
    # Make figure
    if figfile is not None:
        specfigure(figfile,spec,model,out,verbose=verbosity)

    if verbose>0:
        logger.info('dt = %f sec.' % (time.time()-t0))

        
    return out, model
