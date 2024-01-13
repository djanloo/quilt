[![CI](https://github.com/djanloo/quilt/actions/workflows/ci.yml/badge.svg)](https://github.com/djanloo/quilt/actions/workflows/ci.yml)

# quilt

A multiscale neural simulator.

## Installation
- on `Linux`:
  - install boost: ```sudo apt-get install libboost-all-dev```
  - install requirements: ```pip install -r requirements.txt```
  - to build the Python interface run: ```make```
  - to build just the C++ code run: ```make quilt.exe```
- on `Windows`: for now only WSL was tested. Install it and follow the instructions for Linux.
- on `MacOs`: TODO


By default the code is built inplace, so be sure to have this folder in your PYTHONPATH
```
export PYTHONPATH=$(path/to/quilt):$PYTHONPATH
```

## Tests and debug
The `tests` folder contains mainly tests for the Cython/Python interface.

Memory and performance profiling were carried out using respectivley `massif` and `callgrind` from the `valgrind` suite.
