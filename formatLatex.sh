# Backslash in sed is represented by THREE backslashes

cd fdm-devito-notebooks

tp="\\\tp"
newTp="\\\thinspace ."
half="\\\half"
newHalf="\\\frac{1}{2}"
real="\\\Real"
newReal="\\\mathbb{R}"
normalVector="\\\normalvec"
newNormalVector="\\\boldsymbol{n}"

# Set I
setbx="\\\setb{\\\Ix}"
newSetbx="\\\mathcal{I}_x^0"

setex="\\\sete{\\\Ix}"
newSetex="\\\mathcal{I}_x^{-1}"

setlx="\\\setl{\\\Ix}"
newSetlx="\\\mathcal{I}_x^{-}"

setrx="\\\setr{\\\Ix}"
newSetrx="\\\mathcal{I}_x^{+}"

setix="\\\seti{\\\Ix}"
newSetix="\\\mathcal{I}_x^i"

ix="\\\Ix"
newIx="\\\mathcal{I}_x"

setlt="\\\setl{\\\It}"
newSetlt="\\\mathcal{I}_t^{-}"

it="\\\It"
newIt="\\\mathcal{I}_t"

CHAPTERS=("advec" "diffu" "formulas" "nonlin" "softeng2" "trunc" "vib" "wave")

for i in "${CHAPTERS[@]}"; do
  cd $i
  for j in *.ipynb; do
    # replace \tp and account for other places it might get replaced
    sed -i -e "s/\\\tp/\\\thinspace ./g" $j
    sed -i -e "s/outhinspace .uts/outputs/g" $j
    sed -i -e "s/outhinspace .ut_type/output_type/g" $j
    sed -i -e "s/mathinspace .lotlib/matplotlib/g" $j
    sed -i -e "s/htthinspace .s/https/g" $j

    sed -i -e "s/$normalVector/$newNormalVector/g" $j
    sed -i -e "s/$half/$newHalf/g" $j
    sed -i -e "s/$real/$newReal/g" $j

    # I
    # sed -i -e "s/$setbx/$newSetbx/g" $j
    # sed -i -e "s/$setex/$newSetex/g" $j
    # sed -i -e "s/$setlx/$newSetlx/g" $j
    # sed -i -e "s/$setrx/$newSetrx/g" $j
    # sed -i -e "s/$setix/$newSetix/g" $j
    # sed -i -e "s/$ix/$newIx/g" $j
    # sed -i -e "s/$setlt/$newSetlt/g" $j
    # sed -i -e "s/$it/$newIt/g" $j


  done
  rm -rf *.ipynb-e
  cd ..
done