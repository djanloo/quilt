[![CI](https://github.com/djanloo/quilt/actions/workflows/ci.yml/badge.svg)](https://github.com/djanloo/quilt/actions/workflows/ci.yml)

# quilt

A multiscale neural simulator.

Requires boost. On linux
```
sudo apt-get install libboost-all-dev
```
on MacOs
```
brew install boost
```

Build running 
```
make
```
in the main folder. By default the code is built inplace, so be sure to have this folder in your PYTHONPATH
```
export PYTHONPATH=$(pwd):$PYTHONPATH
```
