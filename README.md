![The machine](https://raw.githubusercontent.com/chrislemke/feature-reviser/master/assets/machine.png)

# feature-reviser

[![tests](https://img.shields.io/github/workflow/status/chrislemke/feature-reviser/testing?label=tests&logo=github)](https://github.com/chrislemke/feature-reviser/actions/workflows/testing.yml)
[![build](https://img.shields.io/github/workflow/status/chrislemke/feature-reviser/deploy_package?logo=github)](https://github.com/chrislemke/feature-reviser/actions/workflows/deploy_package.yml)
[![python version](https://img.shields.io/pypi/pyversions/feature-reviser?logo=python&logoColor=yellow)](https://www.python.org/)
[![release](https://img.shields.io/github/v/release/chrislemke/feature-reviser?include_prereleases)](https://github.com/chrislemke/feature-reviser/releases)
[![pypi](https://img.shields.io/pypi/v/feature-reviser)](https://pypi.org/project/feature-reviser/)
[![license](https://img.shields.io/github/license/chrislemke/feature-reviser)](https://github.com/chrislemke/feature-reviser/blob/main/LICENSE)
## Introduction
The feature-reviser makes it easier to find the right features for a classifier.
After creating different features, the question often arises whether they improve or worsen the performance of the classifier and thus have a direct positive or negative influence on the prediction. This project is intended to simplify the selection of features and at the same time contribute to simplifying and automating the entire processing process.

Additionally, this project also contains feature engineering steps, which can be used in a [Scikit-Learn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Check out the [`custom_transformer`](https://github.com/chrislemke/feature-reviser/blob/main/feature_reviser/transformer/custom_transformer.py) module for more information.

## Installation
If you are using [Poetry](https://python-poetry.org/), you can install the package with the following command:
```bash
poetry add feature-reviser
```
If you are using [pip](https://pypi.org/project/pip/), you can install the package with the following command:
```bash
pip install feature-reviser
```

## installing dependencies
With [Poetry](https://python-poetry.org/):
```bash
poetry install
```
With [pip](https://pypi.org/project/pip/):
```bash
pip install -r requirements.txt
```

## Further information
For further information, please refer to the [documentation](https://chrislemke.github.io/feature-reviser/).
