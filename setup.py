#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from sklearn._build_utils import get_blas_info

# Set this to True to enable building extensions using Cython. Set it to False·
# to build extensions from the C file (that was previously generated using·
# Cython). Set it to 'auto' to build with Cython if available, otherwise from·
# the C file.
USE_CYTHON = 'auto'

# If we are in a release, we always never use Cython directly
IS_RELEASE = os.path.exists('PKG-INFO')
if IS_RELEASE:
    USE_CYTHON = False

# If we do want to use Cython, we double check if it is available
if USE_CYTHON:
    try:
        from Cython.Build import cythonize
    except ImportError:
        if USE_CYTHON == 'auto':
            USE_CYTHON = False
        else:
            raise


def configuration():
    config = Configuration('gensvm', '', None)

    cblas_libs, blas_info = get_blas_info()
    if os.name == 'posix':
        cblas_libs.append('m')

    # Wrapper code in Cython uses the .pyx extension if we want to USE_CYTHON, 
    # otherwise it ends in .c.
    wrappers = [
            os.path.join('src', 'wrapper.pyx'),
            ]
    if not USE_CYTHON:
        wrappers = [os.path.splitext(w)[0] + '.c' for w in wrappers]

    # Sources include the C/Cython code from the wrapper and the source code of 
    # the C library
    gensvm_sources = wrappers[:]
    gensvm_sources.append([
            os.path.join('src', 'gensvm', 'src', '*.c'),
            ])

    # Dependencies are the header files of the C library and any potential 
    # helper code between the library and the Cython code
    gensvm_depends = [
            os.path.join('src', 'gensvm', 'include', '*.h'),
            os.path.join('src', 'gensvm', 'gensvm_helper.c')
            ]

    config.add_extension('wrapper',
            sources=gensvm_sources,
            libraries=cblas_libs,
            include_dirs=[
                os.path.join('src', 'gensvm'),
                os.path.join('src', 'gensvm', 'include'),
                numpy.get_include(),
                blas_info.pop('include_dirs', [])],
            extra_compile_args=blas_info.pop('extra_compile_args', []),
            depends=gensvm_depends,
            **blas_info)

    # Cythonize if necessary
    if USE_CYTHON:
        config.ext_modules = cythonize(config.ext_modules)

    return config


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if __name__ == '__main__':

    version = re.search("__version__ = '([^']+)'", 
            open('gensvm/__init__.py').read()).group(1)

    attr = configuration().todict()

    attr['description'] = 'Python package for the GenSVM classifier'
    attr['long_description'] = read('README.rst')
    attr['packages'] = ['gensvm']
    attr['url'] = "https://github.com/GjjvdBurg/PyGenSVM"
    attr['author'] = "G.J.J. van den Burg"
    attr['author_email'] = "gertjanvandenburg@gmail.com"
    attr['license'] = 'GPL v2'
    attr['install_requires'] = ['scikit-learn', 'numpy']

    setup(**attr)
