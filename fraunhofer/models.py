#!/usr/bin/env python

"""MODELS.PY - Programs to deal with model atmospheres

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210201'  # yyyymmdd

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

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


cspeed = 2.99792458e5  # speed of light in km/s


kpath = '/Users/nidever/synspec/winter2017/odfnew/'

# mkmod.pro  make model using interpolation
# mkmadaf.pro   create a composition/continuum opacity file

def read_kurucz(teff,logg,metal,mtype='odfnew'):
    """ Read a Kurucz model from the large grid."""
    #kpath = 'odfnew/'

    s1 = 'a'
    if metal>=0:
        s2 = 'p'
    else:
        s2 = 'm'
    s3 = '%02i' % abs(metal*10)

    if mtype=='old':
        s4 = 'k2.dat'
    elif mtype=='alpha':
        s4 = 'ak2odfnew.dat'
    else:
        s4 = 'k2odfnew.dat'

    filename = kpath+s1+s2+s3+s4

    teffstring = '%7.0f' % teff   # string(teff,format='(f7.0)')
    loggstring = '%8.5f' % logg   # string(logg,format='(f8.5)')
    header = []

    with open(filename,'r') as fil:
        line = fil.readline()
        while (line != '') and (line.find(teffstring) == -1) and (line.find(loggstring) == -1):
            line = fil.readline()

        while (line.find('READ') == -1):
            header.append(line.rstrip())
            line = fil.readline()
        header.append(line.rstrip())

        po = line.find('RHOX')-4
        ntau = int(line[po:po+4].strip())
        if ((ntau == 64 and mtype == 'old') or (ntau == 72)):
            if mtype == 'old':
                model = np.zeros((7,ntau),dtype=np.float64)
            else:
                model = np.zeros((10,ntau),dtype=np.float64)                
        else:
            print('% RD_KMOD: trouble! ntau and type do not match!')
            print('% RD_KMOD: or ntau is neither 64 nor 72')

        for i in range(ntau):
            line = fil.readline()
            model[:,i] = np.array(line.rstrip().split(),dtype=np.float64)
        tail1 = fil.readline().rstrip()
        tail2 = fil.readline().rstrip()
        tail = [tail1,tail2]

        
    return model, header, tail




def mkmod(teff,logg,metal,outfile,ntau=None,mtype='odfnew'):
    """
	Extracts and if necessary interpolates (linearly) a kurucz model 
	from his grid.
	The routine is intended for stars cooler than 10000 K.
	The grid was ftp'ed from CCP7.

	IN: teff	- float - Effective temperature (K)
	    logg	- float - log(g) log_10 of the gravity (cm s-2)
	    metal	- float - [Fe/H] = log N(Fe)/N(H) -  log N(Fe)/N(H)[Sun]
	
	OUT: outfile 	- string - name for the output file

	KEYWORD: ntau	- returns the number of depth points in the output model

	         type  - by default, the k2odfnew grid is used ('type'
				is internally set to 'odfnew') but this
				keyword can be also set to 'old' or 'alpha'
				to use the old models from CCP7, or the 
				ak2odfnew models ([alpha/Fe]=+0.4),respectively.
	

	C. Allende Prieto, UT, May 1999
			   bug fixed, UT, Aug 1999
			   bug fixed to avoid rounoff errors, keyword ntau added
					UT, April 2005
			   bug fixed, assignment of the right tauscale to each
				model (deltaT<1%), UT, March 2006
			   odfnew grids (type keyword), April 2006

    """

    # Constants
    h = 6.626176e-27 # erg s
    c = 299792458e2  # cm s-1
    k = 1.380662e-16 # erg K-1 
    R = 1.097373177e-3 # A-1
    e = 1.6021892e-19 # C
    mn = 1.6749543e-24 # gr
    HIP = 13.60e0

    availteff = np.arange(27)*250+3500.0
    availlogg = np.arange(11)*.5+0.
    availmetal = np.arange(7)*0.5-2.5

    if mtype is None:
        mtype='odfnew'
    if mtype == 'old':
        availmetal = np.arange(13)*0.5-5.0

    if mtype == 'old':
        ntau = 64
    else:
        ntau = 72

    if mtype == 'odfnew' and teff > 10000:
        avail = Table.read(kpath+'tefflogg.txt',format='ascii')
        avail['col1'].name = 'teff'
        avail['col2'].name = 'logg'
        v1,nv1 = dln.where(abs(avail['teff']-teff) < 0.1 and abs(avail['logg']-logg) <= 0.001)
        v2 = v1
        v3,nv3 = dln.where(abs(availmetal-metal) <= 0.001)
    else:
        v1,nv1 = dln.where(abs(availteff-teff) <= .1)
        v2,nv2 = dln.where(abs(availlogg-logg) <= 0.001)
        v3,nv3 = dln.where(abs(availmetal-metal) <= 0.001)

        if (teff <= max(availteff) and teff >= min(availteff) and logg <= max(availlogg) and logg <= min(availlogg) and metal >= min(availmetal) and metal <= max(availmetal)):
            
            if (nv1>0 and nv2>0 and nv3>0):
                # Direct extraction of the model
                teff = availteff[v1[0]]
                logg = availlogg[v2[0]]
                metal = availmetal[v3[0]]
                model,header,tail = read_kurucz(teff,logg,metal,mtype=mtype)
                ntau = len(model[0,:])
	
            else:
                import pdb; pdb.set_trace()
                model = model_interp(teff,logg,metal)

                

        else:
            print('% KMOD:  The requested values of ([Fe/H],logg,Teff) fall outside')
            print('% KMOD:  the boundaries of the grid.')
            print('% KMOD:  Temperatures higher that 10000 K can be reached, by modifying rd_kmod.')


                
    ## writing the outputfile
    #openw,u,outfile
    #for i in range(len(header)):
    #    printf,u,header(i)
    #if type == 'old':
    #    for i in range(ntau):
    #        print(model[:,i],format='(E15.8,x,f8.1,5(x,E9.3))')
    #else:
    #    for i in range(ntau):
    #        print(model[:,i],format='(E15.8,x,f8.1,8(x,E9.3))')

    #for i in range(len(tail)):
    #    printf,u,tail(i)

    return model, header, tail


                
                

def model_interp(teff,logg,metal):
                
    # Linear Interpolation 
    teffimif = max(np.where(availteff <= teff))     # immediately inferior Teff
    loggimif = max(np.where(availlogg <= logg))     # immediately inferior logg
    metalimif = max(np.where(availmetal <= metal))  #immediately inferior [Fe/H]
    teffimsu = min(np.where(availteff >= teff))     # immediately superior Teff
    loggimsu = min(np.where(availlogg >= logg))     # immediately superior logg
    metalimsu = min(np.where(availmetal >= metal))  #immediately superior [Fe/H]
	
    if mtype == 'old':
        ncols = 7
    else:
        ncols = 10
	
    grid = np.zeros((2,2,2,ncols),dtype=np.float64)
    tm1 = availteff[teffimif]
    tp1 = availteff[teffimsu]
    lm1 = availlogg[loggimif]
    lp1 = availlogg[loggimsu]
    mm1 = availmetal[metalimif]
    mp1 = availmetal[metalimsu]

    if (tp1 != tm1):
        mapteff = (teff-tm1)/(tp1-tm1)
    else:
        mapteff = 0.5
    if (lp1 != lm1):
        maplogg = (logg-lm1)/(lp1-lm1)
    else:
        maplogg = 0.5
    if (mp1 != mm1):
        mapmetal = (metal-mm1)/(mp1-mm1)
    else:
        mapmetal = 0.5

    # Reading the corresponding models
    
    for i in range(8):
        if i == 1: model,header,tail = read_kurucz(tm1,lm1,mm1,mtype=mtype)
        if i == 2: model,h,t = read_kurucz(tm1,lm1,mp1,mtype=mtype)
        if i == 3: model,h,t = read_kurucz(tm1,lp1,mm1,mtype=mtype)
        if i == 4: model,h,t = read_kurucz(tm1,lp1,mp1,mtype=mtype)
        if i == 5: model,h,t = read_kurucz(tp1,lm1,mm1,mtype=mtype)
        if i == 6: model,h,t = read_kurucz(tp1,lm1,mp1,mtype=mtype)
        if i == 7: model,h,t = read_kurucz(tp1,lp1,mm1,mtype=mtype)
        if i == 8: model,h,t = read_kurucz(tp1,lp1,mp1,mtype=mtype)

        if (len(model[0,:]) > ntau):
            m2 = np.zeros((ncols,ntau),dtype=np.float64)
            m2[0,:] = interpol(model[0,:],ntau)
            for j in range(ncols):
                m2[j,:] = interpol(model[j,:],model[0,:],m2[0,:])
            model = m2
	    # getting the tauross scale
            rhox = model[0,:]
            kappaross = model[4,:]
            tauross = np.zeros(ntau,dtype=np.float64)
            tauross[0] = rhox[0]*kappaross[0]
            for ii in np.arange(ntau-1)+1:
                tauross[ii] = trapz(rhox[0:ii],kappaross[0:ii])

            if case==1:
                model1 = model 
                tauross1 = tauross
            elif case==2:
                model2 = model
                tauross2 = tauross
            elif case==3:
                model3 = model 
                tauross3 = tauross
            elif case==4:
                model4 = model 
                tauross4 = tauross
            elif case==5:
                model5 = model 
                tauross5 = tauross
            elif case==6:
                model6 = model 
                tauross6 = tauross
            elif case==7:
                model7 = model 
                tauross7 = tauross
            elif case==8:
                model8 = model 
                tauross8 = tauross
            else:
                print('% KMOD: i should be 1--8!')

        model = np.zeros((ncols,ntau),dtype=np.float64)  # cleaning up for re-using the matrix

        # defining the  mass (RHOX#gr cm-2) sampling 
        tauross = tauross1       # re-using the vector tauross
        bot_tauross = min([tauross1[ntau-1],tauross2[ntau-1],
                           tauross3[ntau-1],tauross4[ntau-1],
                           tauross5[ntau-1],tauross6[ntau-1],
                           tauross7[ntau-1],tauross8[ntau-1]])
        top_tauross = max([tauross1[0],tauross2[0],tauross3[0],
                           tauross4[0],tauross5[0],tauross6[0],
                           tauross7[0],tauross8[0]])
        g = np.where(tauross >= top_tauross and tauross <= bot_tauross)
        ntau = len(g)
        tauross_new = interpol(tauross[g])

        # let's interpolate for every depth
        for i in range(ntau):
            for j in range(ncols):
                grid[0,0,0,j] = interpol(model1[j,1:ntau-1],tauross1[1:ntau-1],tauross_new[i])
                grid[0,0,1,j] = interpol(model2[j,1:ntau-1],tauross2[1:ntau-1],tauross_new[i])
                grid[0,1,0,j] = interpol(model3[j,1:ntau-1],tauross3[1:ntau-1],tauross_new[i])
                grid[0,1,1,j] = interpol(model4[j,1:ntau-1],tauross4[1:ntau-1],tauross_new[i])
                grid[1,0,0,j] = interpol(model5[j,1:ntau-1],tauross5[1:ntau-1],tauross_new[i])
                grid[1,0,1,j] = interpol(model6[j,1:ntau-1],tauross6[1:ntau-1],tauross_new[i])
                grid[1,1,0,j] = interpol(model7[j,1:ntau-1],tauross7[1:ntau-1],tauross_new[i])
                grid[1,1,1,j] = interpol(model8[j,1:ntau-1],tauross8[1:ntau-1],tauross_new[i])
                model[j,i] = interpolate(grid[:,:,:,j],mapteff,maplogg,mapmetal)

                for j in range(ncols):
                    model[j,0] = model[j,1]*0.999


        # editing the header
        tmpstr = header[0]
        strput,tmpstr,string(teff,format='(f7.0)'),5
        strput,tmpstr,string(logg,format='(f8.5)'),21
        header[0] = tmpstr
        tmpstr1 = header[1]
        tmpstr2 = header[4]
        if (metal < 0.0):
            if type == 'old':
                strput,tmpstr1,string(abs(metal),format='("-",f3.1)'),18
            else:
                strput,tmpstr1,string(abs(metal),format='("-",f3.1)'),8
            strput,tmpstr2,string(10^metal,format='(f9.5)'),16
        else:
            if type == 'old':
                strput,tmpstr1,string(abs(metal),format='("+",f3.1)'),18
            else:
                strput,tmpstr1,string(abs(metal),format='("+",f3.1)'),8
            strput,tmpstr2,string(10^metal,format='(f9.5)'),16
        header[1] = tmpstr1
        header[4] = tmpstr2
        tmpstr = header[22]
        strput,tmpstr,string(ntau,format='(i2)'),11
        header[22] = tmpstr

    return model, header, tail


  
