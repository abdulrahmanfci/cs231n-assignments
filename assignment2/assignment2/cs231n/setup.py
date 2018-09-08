from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
'''
import pyximport
pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
                              "include_dirs":numpy.get_include()},
                  reload_support=True)
'''
extensions = [
  Extension('im2col_cython', ['im2col_cython.pyx'],
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)
