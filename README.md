[![CI](https://github.com/djanloo/quilt/actions/workflows/ci.yml/badge.svg)](https://github.com/djanloo/quilt/actions/workflows/ci.yml)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/djanloo/quilt/deploy_docs.yml?label=docs)](https://djanloo.github.io/quilt/)

# quilt

A multiscale neural simulator.

## Installation
A virtual environment is highly recommended. 
- on `Linux`:
  - install boost: ```sudo apt-get install libboost-all-dev```
  - update submodules: ```git submodule init && git submodule update```
  - install using pip ```pip install .```
- on `Windows`: for now only WSL was tested. [Install it](https://learn.microsoft.com/en-us/windows/wsl/install) and follow the instructions for Linux.
- on `MacOs`: TODO

### dev installation
Be sure to have your virtual environment set.
For a faster build&install pipeline use the makefile:
- `pip install .[dev]`
- `make`

This builds the code and installs it in the virtual environment.

## Tests and optimization
The `tests` folder contains mainly tests for the Cython/Python interface.

Memory and performance profiling were carried out using respectivley `massif` and `callgrind` from the `valgrind` suite, analysed with the linux `kcachegrind` and `massif-visualizer` tools.
