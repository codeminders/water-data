from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'imputer',
    ext_modules = cythonize("imputation.pyx"),
)
