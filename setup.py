#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from sklearn._build_utils import get_blas_info, maybe_cythonize_extensions


def configuration(parent_package='', top_path=None):
    config = Configuration('gensvm', parent_package, top_path)

    # gensvm module
    cblas_libs, blas_info = get_blas_info()
    if os.name == 'posix':
        cblas_libs.append('m')

    gensvm_sources = [
            os.path.join('gensvm', 'pyx_gensvm.pyx'),
            os.path.join('gensvm', 'src', 'gensvm', 'src', '*.c'),
            ]

    gensvm_depends = [
            os.path.join('gensvm', 'src', 'gensvm', 'include', '*.h'),
            os.path.join('gensvm', 'src', 'gensvm', 'gensvm_helper.c')
            ]

    config.add_extension('pyx_gensvm',
            sources=gensvm_sources,
            libraries=cblas_libs,
            include_dirs=[
                os.path.join('gensvm', 'src', 'gensvm'),
                os.path.join('gensvm', 'src', 'gensvm', 'include'),
                numpy.get_include(),
                blas_info.pop('include_dirs', [])],
            extra_compile_args=blas_info.pop('extra_compile_args', []),
            depends=gensvm_depends,
            **blas_info)
    # end gensvm module

    maybe_cythonize_extensions(top_path, config)

    return config


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if __name__ == '__main__':

    version = re.search("__version__ = '([^']+)'", 
            open('gensvm/__init__.py').read()).group(1)

    attr = configuration(top_path='').todict()

    attr['description'] = 'Python package for the GenSVM classifier'
    attr['long_description'] = read('README.rst')
    attr['packages'] = ['gensvm']
    attr['url'] = "https://github.com/GjjvdBurg/PyGenSVM"
    attr['author'] = "G.J.J. van den Burg"
    attr['author_email'] = "gertjanvandenburg@gmail.com"
    attr['license'] = 'GPL v2'
    attr['install_requires'] = ['scikit-learn', 'numpy']

    setup(**attr)
