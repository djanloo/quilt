[![CI](https://github.com/djanloo/quilt/actions/workflows/ci.yml/badge.svg)](https://github.com/djanloo/quilt/actions/workflows/ci.yml)

# quilt

A multiscale neural simulator.

Requires boost:
```
sudo apt-get install libboost-all-dev
brew install boost
```

To build just the C++ code run:
```
make quilt.exe
```
to build the Python interface too run:
```
make
```


By default the code is built inplace, so be sure to have this folder in your PYTHONPATH
```
export PYTHONPATH=$(pwd):$PYTHONPATH
```
