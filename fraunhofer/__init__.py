__all__ = ["specfit"]


from doppler.spec1d import Spec1D

# Add custom readers here:
from . import reader
# >>>from mymodule import myreader
# >>>reader._readers['myreader'] = myreader
# You can also do this in your own code.

def read(filename=None,format=None):
    return reader.read(filename=filename,format=None)

def fit(*args,**kwargs):
    return specfit.fit(*args,**kwargs)
