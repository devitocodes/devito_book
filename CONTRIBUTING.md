# Contributing to the Devito Book

We welcome third-party contributions, and we would love you to become an active contributor!

Software contributions are made via GitHub pull requests to https://github.com/devitocodes/devito_book. If you are planning a large contribution, we encourage you to engage with us frequently to ensure that your effort is well-directed. See below for more details.

The Devito Book is distributed under the CC BY 4.0 License, https://github.com/devitocodes/devito_book/blob/master/LICENSE.md. The act of submitting a pull request or patch (with or without an explicit Signed-off-by tag) will be understood as an affirmation of the following:

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

### Reporting issues

There are several options:
* Talk to us. You can join our Slack team via this [link](https://devitocodes.slack.com/join/shared_invite/zt-gtd2yxj9-Y31YKk_7lr9AwfXeL2iMFg#/), and join the 'book' channel [here](https://devitocodes.slack.com/archives/C0182SV07NU).
* File an issue on [our GitHub page](https://github.com/devitocodes/devito_book/issues).

### Making changes

First of all, read our [code of conduct](https://github.com/devitocodes/devito_book/blob/master/CODE_OF_CONDUCT.md) and make sure you agree with it.

The protocol to propose a patch is:
* [Recommended, but not compulsory] Talk to us on Slack about what you're trying to do. There is a great chance we can support you.
* As soon as you know which notebook you would like to work on, [fork](https://help.github.com/articles/fork-a-repo/) the Devito Book.
* Create a branch with a suitable name - we suggest the format `<github-username>-<notebook-name>`
* Work on your notebook in `fdm-devito-notebooks`. Commit your changes as small logical units.
* Run the tests for the section you have been working on. There are two things to test for each notebook file:
  1. **Compilation**: Use `pytest` and the `nbval` plugin to check that each of the code cells in your Jupyter Notebook compiles. Running `py.test --nbval <notebook_name>.ipynb` locally should show you the results of these tests (see [`jupyter-notebooks` workflow](https://github.com/devitocodes/devito_book/blob/master/.github/workflows/jupyter-notebooks.yml))
  2. **Verification**: (If applicable) Use `pytest` to run the existing tests in the `.py` files you have been editing and loading functions into the notebook from. Running `py.test -s -v <file_name>.py` locally should show you the results of these tests (see [`verification` workflow](https://github.com/devitocodes/devito_book/blob/master/.github/workflows/verification.yml))
* Add the relevant files you have just tested to the [`jupyter-notebooks`](https://github.com/devitocodes/devito_book/blob/master/.github/workflows/jupyter-notebooks.yml) and [`verification`](https://github.com/devitocodes/devito_book/blob/master/.github/workflows/verification.yml) workflow files on your branch

Skip to **Submitting a Pull Request** if you are not submitting a completed notebook.
* Once all the relevant tests are passing, copy your notebook from `fdm-devito-notebooks` to the relevant chapter in `fdm-jupyter-book/notebooks`, and edit the `_toc.yml` (table of contents) file to include your file. Make sure it's in the right order by checking the readme for that chapter in `fdm-devito-notebooks`
* If your notebook was the last remaining notebook for that chapter of the book, congrats! You can delete the placeholder file for your chapter and remove its reference from `_toc.yml`. Otherwise, just update the placeholder file to no longer reference what you just worked on (since it's now complete). The placeholder files are named as `<chapter-name>_placeholder.md`

#### Submitting a Pull Request 
* Push everything to your Devito Book fork.
* Submit a Pull Request on our repository.
* Wait for us to provide feedback.
