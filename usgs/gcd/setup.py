from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'gcd',
    ext_modules = cythonize("gcd.pyx", working="build"),
    include_dirs = [numpy.get_include()]
)
