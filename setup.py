#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from distutils.command.sdist import sdist

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

docs_require = ["Sphinx==1.6.5", "sphinx_rtd_theme>=0.4.3", "m2r"]
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


def on_cibw_mac():
    return (
        os.environ.get("CIBUILDWHEEL", "0") == "1"
        and os.environ.get("TRAVIS_OS_NAME", "none") == "osx"
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
        def_macros = blas_info_.get("define_macros", [])
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
        blas_info = get_info("blas_opt", notfound_action=0)
        blas_info = {
            "define_macros": [("NO_ATLAS_INFO", 1), ("HAVE_CBLAS", None)],
            "library_dirs": [
                os.sep.join(
                    [
                        "C:",
                        "cibw",
                        "openblas",
                        "OpenBLAS.0.2.14.1",
                        "lib",
                        "native",
                        "lib",
                    ]
                )
            ],
            "include_dirs": [
                os.sep.join(
                    [
                        "C:",
                        "cibw",
                        "openblas",
                        "OpenBLAS.0.2.14.1",
                        "lib",
                        "native",
                        "include",
                    ]
                )
            ],
            "language": "c",
        }
        return ["libopenblas"], blas_info

    blas_info = get_info("blas_opt", notfound_action=2)
    if (not blas_info) or atlas_not_found(blas_info):
        cblas_libs = ["cblas"]
        blas_info.pop("libraries", None)
    else:
        cblas_libs = blas_info.pop("libraries", [])

    if os.environ.get("TRAVIS_OS_NAME", "none") == "osx":
        libdir = blas_info.get("library_dirs", [])
        libdir = libdir[0] if libdir else None
        if libdir:
            base = os.path.split(libdir)[0]
            blas_info["include_dirs"] = [os.path.join(base, "include")]

    if on_cibw_mac():
        blas_info["include_dirs"] = [
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/"
        ]

    return cblas_libs, blas_info


def get_lapack_info():

    from numpy.distutils.system_info import get_info

    def atlas_not_found(lapack_info_):
        def_macros = lapack_info_.get("define_macros", [])
        for x in def_macros:
            if x[0] == "NO_ATLAS_INFO":
                return True
            if x[0] == "ATLAS_INFO":
                if "None" in x[1]:
                    return True
        return False

    if on_cibw_win():
        lapack_info = get_info("lapack_opt", notfound_action=0)
        lapack_info = {
            "define_macros": [("NO_ATLAS_INFO", 1), ("HAVE_CBLAS", None)],
            "library_dirs": [
                os.sep.join(
                    [
                        "C:",
                        "cibw",
                        "openblas",
                        "OpenBLAS.0.2.14.1",
                        "lib",
                        "native",
                        "lib",
                    ]
                )
            ],
            "include_dirs": [
                os.sep.join(
                    [
                        "C:",
                        "cibw",
                        "openblas",
                        "OpenBLAS.0.2.14.1",
                        "lib",
                        "native",
                        "include",
                    ]
                )
            ],
            "language": "c",
        }
        print("***\nDefined lapack info: %r" % lapack_info)
        return ["libopenblas"], lapack_info

    lapack_info = get_info("lapack_opt", notfound_action=2)

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


class mysdist(sdist):
    READMES = ("README", "README.txt", "README.rst", "README.md")


if __name__ == "__main__":
    check_requirements()

    here = os.path.abspath(os.path.dirname(__file__))
    about = {}
    if not VERSION:
        project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
        with open(os.path.join(here, project_slug, "__version__.py")) as fp:
            exec(fp.read(), about)
    else:
        about["__version__"] = VERSION

    try:
        with io.open(os.path.join(here, "README.md"), encoding="utf-8") as fp:
            long_description = "\n" + fp.read()
    except FileNotFoundError:
        long_description = DESCRIPTION

    attr = configuration().todict()

    attr["version"] = about["__version__"]
    attr["description"] = DESCRIPTION
    attr["long_description"] = long_description
    attr["long_description_content_type"] = "text/markdown"
    attr["packages"] = [NAME]
    attr["url"] = URL
    attr["author"] = AUTHOR
    attr["author_email"] = EMAIL
    attr["license"] = LICENSE
    attr["install_requires"] = REQUIRED
    attr["extras_require"] = EXTRAS
    attr["cmdclass"] = {"sdist": mysdist}

    from numpy.distutils.core import setup

    setup(**attr)
