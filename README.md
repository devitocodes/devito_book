[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[![CC BY 4.0][cc-by-shield]][cc-by]
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/devitocodes/devito_book/master)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/devitocodes/devito_book/tree/master/)
![Jupyter Notebooks](https://github.com/devitocodes/devito_book/workflows/Jupyter%20Notebooks/badge.svg)
![Verification](https://github.com/devitocodes/devito_book/workflows/Verification/badge.svg)
![Deploy Jupyter Book](https://github.com/devitocodes/devito_book/workflows/Deploy%20Jupyter%20Book/badge.svg)
[![Slack Status](https://img.shields.io/badge/chat-on%20slack-%2336C5F0)](https://devitocodes.slack.com/join/shared_invite/zt-gtd2yxj9-Y31YKk_7lr9AwfXeL2iMFg#/)


## The Devito Book

The Devito Book is a set of tutorials that focus on the finite difference (FD) method for solving partial differential equations (PDEs), using [Devito](https://github.com/devitocodes/devito). It is largely based on ["Finite Difference Computing with PDEs - A Modern Software Approach"](https://github.com/hplgit/fdm-book) by H. P. Langtangen and S. Linge.

The tutorials are available as a [Jupyter Book](https://www.devitoproject.org/devito_book). They are also available on [MyBinder](https://mybinder.org/v2/gh/devitocodes/devito_book/master) as a set of Jupyter notebooks, under the `fdm-devito-notebooks` directory.

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

### Directory structure

The most important subdirectories in this repository are:

* `fdm-devito-notebooks`, which contains the Jupyter notebooks from the original book with Devito implementations (WIP)
* `fdm-jupyter-book`, which contains the **completed** Jupyter notebooks from `fdm-devito-notebooks`, deployed using GitHub pages [here](https://www.devitoproject.org/devito_book)

### How to use the Devito Book

For an interactive experience with the book, you can run the Jupyter notebooks in your browser using Docker. Run the following command in the terminal:

```
docker-compose up devito_book
```

Alternatively, the non-interactive version of the book is available [here](https://www.devitoproject.org/devito_book).
