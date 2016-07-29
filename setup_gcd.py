from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'gcd',
    ext_modules = cythonize("gcd.pyx"),
)