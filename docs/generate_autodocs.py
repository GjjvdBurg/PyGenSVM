#!/usr/bin/env python

"""
This script manually generates the autodoc RST files for the classes we want to 
document. By doing this, we can generate the documentation on Read The Docs 
(RTD).  If we try to use vanilla autodoc, we run into the problem that a 
working Blas installation is necessary to install the GenSVM python package and 
this is not available in the RTD VM.

Author: Gertjan van den Burg

"""

import os

from docutils.statemachine import StringList, ViewList

from sphinx.ext.autodoc import (AutoDirective, ClassDocumenter, Options, 
        FunctionDocumenter)
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

BASE_DIR = '/home/gertjan/Dropbox/phd/research/msvm/python/start_here/'
DOCDIR = os.path.join(BASE_DIR, 'gensvm', 'docs')

CLASSES = [
        'GenSVMGridSearchCV',
        'GenSVM'
        ]

FUNCTIONS = [
        'load_default_grid'
        ]

FULL_NAMES = {
        'GenSVM': 'gensvm.core.GenSVM',
        'GenSVMGridSearchCV': 'gensvm.gridsearch.GenSVMGridSearchCV',
        'load_default_grid': 'gensvm.gridsearch.load_default_grid'
        }

OUTPUT_FILES = {
        'GenSVMGridSearchCV': os.path.join(DOCDIR, 'cls_gridsearch.rst'),
        'GenSVM': os.path.join(DOCDIR, 'cls_gensvm.rst'),
        'load_default_grid': os.path.join(DOCDIR, 'auto_functions.rst')
        }


def load_app():
    srcdir = DOCDIR[:]
    confdir = DOCDIR[:]
    outdir = os.path.join(BASE_DIR, 'gensvm_docs', 'html')
    doctreedir = os.path.join(BASE_DIR, 'gensvm_docs', 'doctrees')
    buildername = 'html'

    app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername)
    return app


def generate_class_autodoc(app, cls):
    ad = AutoDirective(name='autoclass', arguments=[FULL_NAMES[cls]], 
            options={'noindex': True}, content=StringList([], items=[]), 
            lineno=0, content_offset=1, block_text='', state=None, 
            state_machine=None)

    ad.env = BuildEnvironment(app)
    ad.genopt = Options(noindex=True)
    ad.filename_set = set()
    ad.result = ViewList()

    documenter = ClassDocumenter(ad, ad.arguments[0])
    documenter.generate(all_members=True)

    with open(OUTPUT_FILES[cls], 'w') as fid:
        for line in ad.result:
            fid.write(line + '\n')


def generate_func_autodoc(app, func):
    ad = AutoDirective(name='autofunc', arguments=[FULL_NAMES[func]], 
            options={'noindex': True}, content=StringList([], items=[]), 
            lineno=0, content_offset=1, block_text='', state=None, 
            state_machine=None)

    ad.env = BuildEnvironment(app)
    ad.genopt = Options(noindex=True)
    ad.filename_set = set()
    ad.result = ViewList()

    documenter = FunctionDocumenter(ad, ad.arguments[0])
    documenter.generate(all_members=True)

    with open(OUTPUT_FILES[func], 'a') as fid:
        for line in ad.result:
            fid.write(line + '\n')


def main():
    for of in OUTPUT_FILES:
        fname = OUTPUT_FILES[of]
        os.unlink(fname)
    app = load_app()
    for cls in CLASSES:
        generate_class_autodoc(app, cls)
    for func in FUNCTIONS:
        generate_func_autodoc(app, func)


if __name__ == '__main__':
    main()
