cd fdm-devito-notebooks

tp="\\tp"
newTp="\\thinspace ."
half="\\half"
newHalf="\\frac{1}{2}"
# sed -i -e "s/$tp/$newTp/g" fdm-devito-notebooks/wave/wave1D_fd2.ipynb
# Fix "outputs" string which is also changed by previous command
# sed -i -e "s/outhinspace .uts/outputs/g" fdm-devito-notebooks/wave/wave1D_fd2.ipynb

CHAPTERS=("advec" "diffu" "formulas" "nonlin" "softeng2" "trunc" "vib" "wave")
# CHAPTERS=("wave")

for i in "${CHAPTERS[@]}"; do
  cd $i
  for j in *.ipynb; do
    sed -i -e "s/$tp/$newTp/g" $j
    sed -i -e "s/outhinspace .uts/outputs/g" $j
    sed -i -e "s/outhinspace .ut_type/output_type/g" $j
    sed -i -e "s/$half/$newHalf/g" $j
  done
  rm -rf *.ipynb-e
  cd ..
done