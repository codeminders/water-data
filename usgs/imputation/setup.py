from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'imputer',
    ext_modules = cythonize("imputation.pyx"),
    include_dirs = [numpy.get_include()]
)
