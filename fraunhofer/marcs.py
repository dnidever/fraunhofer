import os
import numpy as np
from glob import glob
from astropy.table import Table
from dlnpyutils import utils as dln

index = Table.read('marcs_index.fits')
data = dln.unpickle('marcs_data.pkl')


def readmarcs(filename):
    """ Read the MARCS file and returns the essentials needed for Korg."""
    if os.path.exists(filename)==False:
        raise ValueError(filename+' NOT FOUND')
    # Load the data
    with open(filename,"r") as f:
        data = f.read()
    # Trim
    lines = np.char.array(data.split('\n'))
    hi, = np.where(lines.find('KappaRoss')>-1)
    lines = lines[0:hi[0]]
    # Get the parameters
    teffind, = np.where(lines.find('Teff')>-1)
    if len(teffind)>0:
        teff = float(lines[teffind[0]].split()[0])
    loggind, = np.where(lines.find('Surface gravity')>-1)
    if len(loggind)>0:
        logg = np.log10(float(lines[loggind[0]].split()[0]))
        # round to closest decimal place
        logg = np.round(logg,decimals=2)
    microind, = np.where(lines.find('Microturbulence')>-1)
    if len(microind):
        vmicro = float(lines[microind[0]].split()[0])
    metalind, = np.where(lines.find('Metallicity')>-1)
    if len(metalind)>0:
        metal = float(lines[metalind[0]].split()[0])
        alpha = float(lines[metalind[0]].split()[1])
    out = dict()
    out['teff'] = teff
    out['logg'] = logg
    out['vmicro'] = vmicro
    out['metal'] = metal
    out['alpha'] = alpha
    out['lines'] = lines

    return out


def convertall():
    """ Load all MARCS models and save to a pickle file."""

    files = glob('/Users/nidever/marcs/mod_z*/s*.mod*')
    nfiles = len(files)
    print(nfiles,' files')
    
    dt = [('index',int),('filename',str,100),('teff',float),('logg',float),('vmicro',float),('metal',float),('alpha',float)]
    tab = np.zeros(nfiles,dtype=np.dtype(dt))
    data = []
    for i in range(nfiles):
        data1 = readmarcs(files[i])
        filename = os.path.basename(files[i])
        tab['index'][i] = i
        tab['filename'][i] = filename
        tab['teff'][i] = data1['teff']
        tab['logg'][i] = data1['logg']
        tab['vmicro'][i] = data1['vmicro']
        tab['metal'][i] = data1['metal']
        tab['alpha'][i] = data1['alpha']
        print(i,data1['teff'],data1['logg'],data1['vmicro'],data1['metal'],data1['alpha'])
        data.append(list(data1['lines']))
    Table(tab).write('/Users/nidever/marcs/marcs_index.fits',overwrite=True)
    dln.pickle('/Users/nidever/marcs/marcs_data.pkl',data)

    import pdb; pdb.set_trace()

def findmodel(teff,logg,metal,vmicro=2.0,alpha=0.0):
    """ Return the MARCS model information for a given set of parameters."""

    ind, = np.where((abs(index['teff']-teff) < 1) & (abs(index['logg']-logg)<0.01) &
                    (abs(index['metal']-metal)<0.01) & (abs(index['vmicro']-vmicro)<0.01) &
                    (abs(index['alpha']-alpha)<0.01))
    if len(ind)==0:
        print('Could not find model for the input parameters')
        return
    lines = data[ind[0]]
    return lines

