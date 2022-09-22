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

## Feature reviser
Finding the best features for your model is hard. In the `feature_selection` part of the project, we try to automate this process to make it a bit easier. This part of the project is still in development and is not yet ready for use. 🚧 If you want to help, you can find more information in the [contributing guide](how_to_contribute.md).

## The transformer module
Data preprocessing often involves similar processes. No matter whether it is a matter of manipulating strings or numbers. [Scikit-learn's pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) implementation makes it easy to structure and sequence such preprocessing processes. To take advantage of this, the [`custom_transformers`](https://github.com/chrislemke/feature-reviser/blob/main/feature_reviser/transformer/custom_transformer.py) module includes a set of transformers that can be easily pipelined to simplify preprocessing. The list of transformers is open and will be extended permanently.

## Contributing
We're all kind of in the same boat. Preprocessing in data science is somehow very individual - every feature is different and must be handled and processed differently. But somehow we all have the same problems: sometimes date columns have to be changed. Sometimes strings have to be formatted, sometimes durations have to be calculated, etc. There is a huge number of preprocessing possibilities but we all use the same tools.

[Scikit-learns pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) help to use formalized functions. So why not also share these so-called transformers with others? This open source project has the goal to collect useful preprocessing pipeline steps. Let us all collect what we used for preprocessing and share it with others. This way we can all benefit from each other's work and save a lot of time. So if you have a preprocessing step that you use regularly, please feel free to contribute it to this project. The idea is that this is not only a toolbox but also an inspiration for what is possible. Maybe you have not thought about this preprocessing step before.

Please check out the [documentation](how_to_contribute.md) on how to contribute to this project.


## Further information
For further information, please refer to the [documentation](https://chrislemke.github.io/feature-reviser/).
