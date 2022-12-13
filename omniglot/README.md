# Grouped Omniglot

Task-Informed Meta-Learning applied to the [Omniglot dataset](https://github.com/brendenlake/omniglot). This dataset must be downloaded into the [data folder](data).

### Getting started

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `timl-omniglot` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate timl-omniglot
```

The main script to train the models is then [`run.py`](run.py), with the model configurations controlled by the [`config`](src/config.py).
