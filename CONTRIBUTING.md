# Contributing to the Devito Book

We welcome third-party contributions, and we would love you to become an active contributor!

Software contributions are made via GitHub pull requests to https://github.com/devitocodes/devito_book. If you are planning a large contribution, we encourage you to engage with us frequently to ensure that your effort is well-directed. See below for more details.

### Reporting issues

There are several options:
* Talk to us. You can join our Slack team via this [link](https://opesci-slackin.now.sh/), and join the 'book' channel [here](https://devitocodes.slack.com/archives/C0182SV07NU).
* File an issue on [our GitHub page](https://github.com/devitocodes/devito_book/issues).

### Making changes

First of all, read our [code of conduct](https://github.com/devitocodes/devito_book/blob/master/CODE_OF_CONDUCT.md) and make sure you agree with it.

The protocol to propose a patch is:
* [Recommended, but not compulsory] Talk to us on Slack about what you're trying to do. There is a great chance we can support you.
* As soon as you know what you need to do, [fork](https://help.github.com/articles/fork-a-repo/) the Devito Book.
* Create a branch with a suitable name.
* Write code following the guidelines below. Commit your changes as small logical units.
* Commit messages should adhere to the format `<tag>: <msg>`, where `<tag>` could be, for example, "wave" (if the commit impacts the waves section of the book), "nonlin", "tests", etc. We may ask you to rebase the commit history if it looks too messy.
* Push everything to your Devito Book fork.
* Submit a Pull Request on our repository.
* Wait for us to provide feedback.

Tip, especially for newcomers: prefer short, self-contained Pull Requests over lengthy, impenetrable, and thus difficult to review, ones.

### Process for working on a notebook

1. Work on your notebook in `fdm-devito-notebooks`. This contains all of the Jupyter notebooks (`.ipynb` files) that make up the book. We recommend you run:

```
jupyter notebook fdm-devito-notebooks
```

and navigate to the notebook you are working on in your browser.

2. Once you are happy that the notebook has been successfully "Devito-fied", submit a pull request as described above.