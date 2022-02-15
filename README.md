# Task-Informed Meta-Learning

This repository contains examples of Task-Informed Meta-Learning ([paper](https://arxiv.org/abs/2202.02124)).

We consider two tasks:

- [Crop Type Classification](crop_classification)
- [Yield Estimation](yield)

Each task acts as its own self-contained codebase - for more details on running the experiments, please check their respective READMEs.

### Getting started

For both tasks, [anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up with an environment, install Anaconda from the link above, and (from either of the directories) run

```bash
conda env create -f environment.yml
```
Once the environment is activated, the main script to train the models is then `deep_learning.py`, with the model configurations controlled by the `config.py` file.

The trained TIML models are available on [Zenodo](https://zenodo.org/record/6089828).
