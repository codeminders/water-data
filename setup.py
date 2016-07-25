from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'cubic solver',
    ext_modules = cythonize("solve_cubic.pyx"),
)
