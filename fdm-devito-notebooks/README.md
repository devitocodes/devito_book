# How to navigate this directory

Every folder inside `fdm-devito-notebooks` refers to a section of the book. The chapters are labelled numerically and the appendices are labelled alphabetically.

The folder for each section contains the Jupyter notebooks (`.ipynb` files) for that section, a readme to explain which notebooks correspond to which chapters of the original book section, and some subdirectories. The most important of these subdirectories are:

* `exer-<section>`: the Python code for that section's exercises
* `fig-<section>`: the images displayed in that section
* `mov-<section>`: the videos displayed in that section
* `src-<section>`: the Python code that makes up that section's tutorials (including tests)

As there are occasionally issues with the rendering of Jupyter notebooks on GitHub, you can view these notebooks using NBViewer [here](https://nbviewer.jupyter.org/github/devitocodes/devito_book/tree/master/fdm-devito-notebooks/).

## Information for contributors

A Jupyter notebook is considered 'complete' when the following is true:

* The Python code segments have been **correctly** re-implemented using Devito - i.e., the tests still pass. (For more on testing, see [here](https://github.com/devitocodes/devito_book/wiki/How-do-I-test-a-notebook-or-set-of-notebooks-with-Devito-implementations%3F))
* The following are all correctly formatted and, if applicable, link to the correct file:
  * LaTeX symbols
  * Images
  * Videos
  * Websites
  * Links to Python files
  * References to other sections of the book (either within the same notebook or to other `.ipynb` files)

For information on how to make sure your notebook is correctly formatted, see [the wiki](https://github.com/devitocodes/devito_book/wiki) for this repository.
