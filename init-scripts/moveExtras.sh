#!/bin/bash

cd ..
cd doc/.src/chapters

CHAPTERS=("advec" "diffu" "formulas" "nonlin" "softeng2" "trunc" "vib" "wave")
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