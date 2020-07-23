# NB: Backslash in sed is represented by THREE backslashes

cd fdm-devito-notebooks

# Array containing custom DocOnce commands
DOCONCE=("\\\tp" "\\\half" "\\\Real" "\\\normalvec")

# Array containing pure LaTeX commands, where
# PURE_LATEX[i] is the pure LaTeX version of DOCONCE[i]
PURE_LATEX=("\\\thinspace ." "\\\frac{1}{2}" "\\\mathbb{R}" "\\\boldsymbol{n}")

# Auxilliary errors as a result of the above formatting and their fixes
AUX_ERRORS=("outhinspace .ut" "athinspace .lot")
AUX_FIXES=("output" "atplot")

# Chapter directory names
CHAPTERS=("advec" "diffu" "formulas" "nonlin" "softeng2" "trunc" "vib" "wave")

# If arrays are different lengths, script is not executed and throws error
if [ ${#DOCONCE[@]} -ne ${#PURE_LATEX[@]} ]; then
  echo "ERROR: DocOnce and pure LaTeX arrays are different lengths"
  exit 1
fi

for chapter in "${CHAPTERS[@]}"; do
  cd $chapter
  # Only replace text in Jupyter notebooks (.ipynb files)
  for notebook in *.ipynb; do
    for index in ${!DOCONCE[@]}; do
      sed -i -e "s/${DOCONCE[index]}/${PURE_LATEX[index]}/g" $notebook
    done
    for index in ${!AUX_ERRORS[@]}; do
      sed -i -e "s/${AUX_ERRORS[index]}/${AUX_FIXES[index]}/g" $notebook
    done
  done
  # Remove extra files generated 
  rm -rf *.ipynb-e
  cd ..
done

echo "Successfully formatted .ipynb files in ${CHAPTERS[@]}"