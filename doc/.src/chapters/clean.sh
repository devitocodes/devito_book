#!/bin/sh -x
# To be run from a chapter subdirectory: bash ../clean.sh
doconce clean

rm -rf .ptex2tex.cfg* newcommands*.tex automake_sphinx.py *~ tmp* _doconce_deb* *-4print.pdf *-4screen.pdf *-sol.pdf latex_figs *.pyc
