#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

# Package meta-data
AUTHOR = "Gertjan van den Burg"
DESCRIPTION = "Generalized Multiclass Support Vector Machines"
EMAIL = "gertjanvandenburg@gmail.com"
LICENSE = "GPLv2"
LICENSE_TROVE = (
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)"
)
NAME = "gensvm"
REQUIRES_PYTHON = ">=2.7"
URL = "https://github.com/GjjvdBurg/PyGenSVM"
VERSION = None

REQUIRED = ["scikit-learn", "numpy"]

docs_require = ["Sphinx==1.6.5", "sphinx_rtd_theme>=0.4.3"]
test_require = []
dev_require = ["Cython"]

EXTRAS = {
    "docs": docs_require,
    "tests": test_require,
    "dev": docs_require + test_require + dev_require,
}

# Set this to True to enable building extensions using Cython. Set it to False·
# to build extensions from the C file (that was previously generated using·
# Cython). Set it to 'auto' to build with Cython if available, otherwise from·
# the C file.
USE_CYTHON = "auto"

# If we are in a release, we never use Cython directly
IS_RELEASE = os.path.exists("PKG-INFO")
if IS_RELEASE:
    USE_CYTHON = False

# If we do want to use Cython, we double check if it is available
if USE_CYTHON:
    try:
        from Cython.Build import cythonize
    except ImportError:
        if USE_CYTHON == "auto":
            USE_CYTHON = False
        else:
            raise

# Try to load setuptools, so that NumPy's distutils module that we use to
# provide the setup() function below comes from the setuptools package. If it
# fails, it'll use distutils' version, which doesn't support installing
# dependencies.
try:
    import setuptools
except ImportError:
    print(
        "Warning: setuptools not found. You may have to install GenSVM's dependencies manually."
    )


def on_cibw_win():
    return (
        os.environ.get("CIBUILDWHEEL", "0") == "1"
        and os.environ.get("TRAVIS_OS_NAME", "none") == "windows"
    )


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
        def_macros = blas_info.get("define_macros", [])
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

    if on_cibw_win():
        blas_info = {
            "define_macros": [("NO_ATLAS_INFO", 1), ("HAVE_CBLAS", None)],
            "libraries": ["openblas"],
            "library_dirs": [
                "/c/cibw/openblas/OpenBLAS.0.2.14.1/lib/native/lib/"
            ],
            "include_dirs": [
                "/c/cibw/openblas/OpenBLAS.0.2.14.1/lib/native/include"
            ],
            "language": "c",
        }
        return ["cblas"], blas_info

    blas_info = get_info("blas_opt", notfound_action=2)
    print("\n\n*** blas_info: \n%r\n\n ***\n\n" % blas_info)
    print(
        "\n\n*** os.environ.get('OPENBLAS') = %r ***\n\n"
        % (os.environ.get("OPENBLAS", None))
    )
    if (not blas_info) or atlas_not_found(blas_info):
        cblas_libs = ["cblas"]
        blas_info.pop("libraries", None)
    else:
        cblas_libs = blas_info.pop("libraries", [])

    return cblas_libs, blas_info


def get_lapack_info():

    from numpy.distutils.system_info import get_info

    def atlas_not_found(lapack_info_):
        def_macros = lapack_info.get("define_macros", [])
        for x in def_macros:
            if x[0] == "NO_ATLAS_INFO":
                return True
            if x[0] == "ATLAS_INFO":
                if "None" in x[1]:
                    return True
        return False

    if on_cibw_win():
        blas_info = {
            "define_macros": [("NO_ATLAS_INFO", 1), ("HAVE_CBLAS", None)],
            "libraries": ["openblas"],
            "library_dirs": [
                "/c/cibw/openblas/OpenBLAS.0.2.14.1/lib/native/lib/"
            ],
            "include_dirs": [
                "/c/cibw/openblas/OpenBLAS.0.2.14.1/lib/native/include"
            ],
            "language": "c",
        }
        return ["cblas"], blas_info

    lapack_info = get_info("lapack_opt", notfound_action=2)
    print("\n\n*** lapack_info: \n%r\n\n ***\n\n" % lapack_info)
    print(
        "\n\n*** os.environ.get('LAPACK') = %r ***\n\n"
        % (os.environ.get("LAPACK", None))
    )
    if (not lapack_info) or atlas_not_found(lapack_info):
        # This is a guess, but seems to work in practice. Need more systems to
        # test this fully.
        lapack_libs = ["lapack"]
        lapack_info.pop("libraries", None)
    else:
        lapack_libs = lapack_info.pop("libraries", [])

    return lapack_libs, lapack_info


def configuration():
    from numpy.distutils.misc_util import Configuration

    config = Configuration("gensvm", "", None)

    cblas_libs, blas_info = _skl_get_blas_info()
    if os.name == "posix":
        cblas_libs.append("m")

    lapack_libs, lapack_info = get_lapack_info()
    if os.name == "posix":
        lapack_libs.append("m")  # unsure if necessary

    # Wrapper code in Cython uses the .pyx extension if we want to USE_CYTHON,
    # otherwise it ends in .c.
    wrapper_extension = "*.pyx" if USE_CYTHON else "*.c"

    # Sources include the C/Cython code from the wrapper and the source code of
    # the C library
    gensvm_sources = [
        os.path.join("gensvm", "cython_wrapper", wrapper_extension),
        os.path.join("src", "gensvm", "src", "*.c"),
    ]

    # Dependencies are the header files of the C library and any potential
    # helper code between the library and the Cython code
    gensvm_depends = [
        os.path.join("src", "gensvm", "include", "*.h"),
        os.path.join("src", "gensvm", "gensvm_helper.c"),
    ]

    from numpy import get_include

    config.add_extension(
        "cython_wrapper.wrapper",
        sources=gensvm_sources,
        libraries=cblas_libs + lapack_libs,
        include_dirs=[
            os.path.join("src", "gensvm"),
            os.path.join("src", "gensvm", "include"),
            get_include(),
            blas_info.pop("include_dirs", []),
        ],
        extra_compile_args=blas_info.pop("extra_compile_args", []),
        depends=gensvm_depends,
        **blas_info
    )

    # Cythonize if necessary
    if USE_CYTHON:
        config.ext_modules = cythonize(config.ext_modules)

    return config


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def check_requirements():
    numpy_instructions = (
        "\n"
        "GenSVM Installation Error:"
        "\n"
        "Numerical Python (NumPy) is not installed on your "
        "system. This package is required for GenSVM. Please install "
        "NumPy using the instructions available here: "
        "https://docs.scipy.org/doc/numpy-1.13.0/user/install.html"
    )

    try:
        import numpy

        numpy_version = numpy.__version__
    except ImportError:
        raise ImportError(numpy_instructions)


def cibuildwheel_windows():
    if not on_cibw_win():
        return
    print("\n*** Preparing GenSVM for CIBuildWheel ***")

    import shutil

    # check if we're executing python in 32bit or 64bit mode
    bits = 64 if sys.maxsize > 2 ** 32 else 32
    bitprefix = "x64" if bits == 64 else "win32"

    basepath = "/c/cibw/openblas/OpenBLAS.0.2.14.1/lib/native"
    dllpath = basepath + "/lib/" + bitprefix + "/libopenblas.dll.a"
    if os.path.exists(dllpath):
        shutil.move(dllpath, basepath + "/lib/")

    os.environ[
        "OPENBLAS"
    ] = "/c/cibw/openblas/OpenBLAS.0.2.14.1/lib/native/lib"
    print(os.environ.get("OPENBLAS", "none"))

    for path, dirs, files in os.walk("/c/cibw/openblas"):
        print(path)
        for f in files:
            print("\t" + f)
    sys.stdout.flush()
    import time

    time.sleep(5)


if __name__ == "__main__":
    check_requirements()
    cibuildwheel_windows()

    version = re.search(
        '__version__ = "([^\']+)"', open("gensvm/__init__.py").read()
    ).group(1)

    attr = configuration().todict()

    attr["version"] = version
    attr["description"] = DESCRIPTION
    attr["long_description"] = read("README.rst")
    attr["packages"] = [NAME]
    attr["url"] = URL
    attr["author"] = AUTHOR
    attr["author_email"] = EMAIL
    attr["license"] = LICENSE
    attr["install_requires"] = REQUIRED
    attr["extras_require"] = EXTRAS

    from numpy.distutils.core import setup

    setup(**attr)
