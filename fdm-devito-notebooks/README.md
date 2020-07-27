# How to navigate this directory

Every folder inside `fdm-devito-notebooks` refers to a section of the book. The chapters are labelled numerically and the appendices are labelled alphabetically.

The folder for each section contains the Jupyter notebooks (`.ipynb` files) that make up the book, as well as some subdirectories. The most important of these are:

* `exer-<section>`: the Python code for that section's exercises
* `fig-<section>`: the images displayed in that section
* `mov-<section>`: the videos displayed in that section
* `src-<section>`: the Python code that makes up that section's tutorials (including tests)

## Information for contributors

A Jupyter notebook is considered 'complete' when the following is true:

* The Python code segments have been **correctly** re-implemented using Devito - i.e., the tests still pass. (For more on testing, see [here](https://github.com/devitocodes/devito_book/wiki/How-do-I-test-a-notebook-or-set-of-notebooks-with-Devito-implementations%3F)
* The following are all correctly formatted and, if applicable, link to the correct file:
  * LaTeX symbols
  * Images
  * Videos
  * Websites
  * Citations
  * References to other sections of the book (either within the same notebook or to other `.ipynb` files)
  
When all of this is complete, the notebook can be copied to the `fdm-jupyter-book` folder on the master branch and deployed to GitHub pages.
