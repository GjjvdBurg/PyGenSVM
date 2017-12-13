#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

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

def _skl_get_blas_info():
    """Copyright notice for this function

    Copyright (c) 2007–2017 The scikit-learn developers.
    All rights reserved.


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

      a. Redistributions of source code must retain the above copyright notice,
         this list of conditions and the following disclaimer.
      b. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
      c. Neither the name of the Scikit-learn Developers  nor the names of
         its contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.

    """
    from numpy.distutils.system_info import get_info

    def atlas_not_found(blas_info_):
        def_macros = blas_info.get('define_macros', [])
        for x in def_macros:
            if x[0] == "NO_ATLAS_INFO":
                # if x[1] != 1 we should have lapack
                # how do we do that now?
                return True
            if x[0] == "ATLAS_INFO":
                if "None" in x[1]:
                    # this one turned up on FreeBSD
                    return True
        return False

    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or atlas_not_found(blas_info):
        cblas_libs = ['cblas']
        blas_info.pop('libraries', None)
    else:
        cblas_libs = blas_info.pop('libraries', [])

    return cblas_libs, blas_info


def configuration():
    from numpy.distutils.misc_util import Configuration
    config = Configuration('gensvm', '', None)

    cblas_libs, blas_info = _skl_get_blas_info()
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

    from numpy import get_include
    config.add_extension('wrapper',
            sources=gensvm_sources,
            libraries=cblas_libs,
            include_dirs=[
                os.path.join('src', 'gensvm'),
                os.path.join('src', 'gensvm', 'include'),
                get_include(),
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


def check_requirements():
    numpy_instructions = ("Numerical Python (NumPy) is not installed on your "
                "system. This package is required for GenSVM. Please install "
                "NumPy using the instructions available here: "
                "https://docs.scipy.org/doc/numpy-1.13.0/user/install.html")

    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        raise ImportError(numpy_instructions)


if __name__ == '__main__':
    check_requirements()

    version = re.search("__version__ = '([^']+)'", 
            open('gensvm/__init__.py').read()).group(1)

    attr = configuration().todict()

    attr['version'] = version
    attr['description'] = 'Python package for the GenSVM classifier'
    attr['long_description'] = read('README.rst')
    attr['packages'] = ['gensvm']
    attr['url'] = "https://github.com/GjjvdBurg/PyGenSVM"
    attr['author'] = "G.J.J. van den Burg"
    attr['author_email'] = "gertjanvandenburg@gmail.com"
    attr['license'] = 'GPL v2'
    attr['install_requires'] = ['scikit-learn', 'numpy', 'scipy']

    from numpy.distutils.core import setup
    setup(**attr)
