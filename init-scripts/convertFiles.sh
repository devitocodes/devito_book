# !/bin/bash
# Script that takes DocOnce files from all chapters (except waves) of FD book and converts into .ipynb format
# DO NOT RUN THIS SCRIPT - WILL REPLACE ALL WORK

cd ..
cd doc/.src/chapters

CHAPTERS=("01_vib" "02_wave" "03_diffu" "04_advec" "05_nonlin" "A_formulas" "B_trunc" "C_softeng2")

TMPS=()

for i in "${CHAPTERS[@]}"; do
  cd $i
  for j in *.do.txt; do
    doconce format ipynb $j --no_abort
  done
  # Move IPython notebooks, dlog files and tar.gz files to fdm-devito-notebooks directory
  mkdir -p ../../../../fdm-devito-notebooks/$i
  mv *.ipynb ../../../../fdm-devito-notebooks/$i
  rm -rf *.dlog
  rm -rf *.tar.gz

  # Add temporary files to TMPS
  while IFS= read -r line; do
      TMPS+=( "$line" )
  done < <( find . -type f -name "*tmp_mako*" -name "*.do.txt" )

  while IFS= read -r line; do
      TMPS+=( "$line" )
  done < <( find . -type f -name "*tmp_preprocess*" -name "*.do.txt" )

  # Delete temporary files
  for j in "${TMPS[@]}"; do
    rm -rf $j
  done
  TMPS=()

  cd ..
done

