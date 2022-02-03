# Yield Estimation

Task-Informed Meta-Learning applied to yield estimation, using the yield dataset first described in [Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data](https://cs.stanford.edu/~ermon/papers/cropyield_AAAI17.pdf). We recreate this dataset using the [PyTorch implementation](https://github.com/gabrieltseng/pycrop-yield-prediction/), rerun with all default arguments.

### Getting started

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `timl-yield` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate timl-yield
```

The main script to train the models is then [`deep_learning.py`](deep_learning.py), with the model configurations controlled by the [`config`](config.py). Running this script will automatically download the data into the [data folder](data) from [Zenodo](https://zenodo.org/record/5948877).
