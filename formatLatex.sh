tp="\\tp"
newTp="\\thinspace ."
sed -i -e "s/$tp/$newTp/g" fdm-devito-notebooks/wave/wave1D_fd2.ipynb
# Fix "outputs" string which is also changed by previous command
sed -i -e "s/outhinspace .uts/outputs/g" fdm-devito-notebooks/wave/wave1D_fd2.ipynb