#!/bin/bash
# Script that takes DocOnce files from Waves chapter of FD method and converts into .ipynb format

mkdir -p jupyter-fdm-book
cd doc/.src/chapters/wave

for i in *.do.txt; do
  doconce format ipynb $i --no_abort
done

# Move IPython notebooks, dlog files and tar.gz files to jupyter-fdm-book directory
mv *.ipynb ../../../../jupyter-fdm-book
mv *.dlog ../../../../jupyter-fdm-book
mv *.tar.gz ../../../../jupyter-fdm-book

# Create an array to store temporary files
TMPS=()
while IFS= read -r line; do
    TMPS+=( "$line" )
done < <( find . -type f -name "*tmp_mako*" -name "*.do.txt" )

while IFS= read -r line; do
    TMPS+=( "$line" )
done < <( find . -type f -name "*tmp_preprocess*" -name "*.do.txt" )

# Delete temporary files
for i in "${TMPS[@]}"; do
  rm -rf $i
done

# cd jupyter-fdm-book
# jupyter notebook wave1D_fd1.ipynb