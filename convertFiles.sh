#!/bin/bash
# Script that takes DocOnce files from Waves chapter of FD method and converts into .ipynb format

mkdir -p jupyter-fdm-book
cd doc/.src/chapters/wave

for i in *.do.txt; do
  doconce format ipynb $i --no_abort
done

mv *.ipynb ../../../../jupyter-fdm-book
