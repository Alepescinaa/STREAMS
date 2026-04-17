## Python utilities for the R package

This folder contains Python code used internally by the R package.\
The script `train.py` implements a Mean Teacher conditional variational autoencoder generating a individual-level posterior distribution for disease onset and onset age. Thr script `inference.py` leverages the model for prediction. It is intended to be called directly in Python.

## 📦 Install Python dependencies

Before using the Python scripts, install the required modules:

``` bash
pip install -r inst/requirements.txt
```

## Check Python version

These two should be the same.

R console:

``` r
system2("python", args = "--version")
```

Terminal:

``` bash
python --version
```

If they don't, check the path used by R:

``` r
system2("python", args = "-c \"import sys; print(sys.executable)\"")
```

Install Python dependencies on this path:

``` bash
'complete path' -m pip install -r requirements.txt
```

## Install Python dependencies from R

Alternatively the following R code can be used to install dependencies:

``` r
library(reticulate)

py <- import("sys")$executable
system2(py, c("-m", "ensurepip", "--default-pip"))
system2(py, c("-m", "pip", "--version"))
system2(py, c("-m", "pip", "install",
              "numpy>=1.20",
              "torch>=1.12",
              "pyarrow>=7.0",
              "scikit-learn>=1.0",
              "tqdm>=4.0",
              "pandas>=1.3"))
```
