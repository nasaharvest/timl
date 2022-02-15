# Crop Classification

Task-Informed Meta-Learning applied to crop classification, using the [cropharvest](https://github.com/nasaharvest/cropharvest/) dataset.

This repository mirrors the structure of the [cropharvest benchmarks](https://github.com/nasaharvest/cropharvest/tree/main/benchmarks) - specifically, we train a meta-model in the [TIML Learner](dl/timl.py), and use the state dictionary saved by the learner in a finetuning loop.

### Getting started

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `timl-classification` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate timl-classification
```

The main script to train the models is then [`deep_learning.py`](deep_learning.py), with the model configurations controlled by the [`config`](config.py)

The trained TIML model is available on [Zenodo](https://zenodo.org/record/6089828).
