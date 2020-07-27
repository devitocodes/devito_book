#!/bin/bash

cd ..
cd doc/.src/chapters

CHAPTERS=("01_vib" "02_wave" "03_diffu" "04_advec" "05_nonlin" "A_formulas" "B_trunc" "C_softeng2")
EXTRAS=("fig" "mov" "src" "exer" "slides")

for i in "${CHAPTERS[@]}"; do
    cd $i
    for j in "${EXTRAS[@]}"; do
        if [ -d "$j-$i" ]; then
            mkdir -p ../../../../fdm-devito-notebooks/$i/"$j-$i"
            cp -r "$j-$i"/ ../../../../fdm-devito-notebooks/$i/"$j-$i"
        fi
    done
    cd ..
done