<h2 align="center">
<img src="https://github.com/djanloo/quilt/blob/main/assets/logo.svg" width="250">
</h2><br>

[![CI](https://github.com/djanloo/quilt/actions/workflows/ci.yml/badge.svg)](https://github.com/djanloo/quilt/actions/workflows/ci.yml)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/djanloo/quilt/deploy_docs.yml?label=docs)](https://djanloo.github.io/quilt/)


https://github.com/user-attachments/assets/bc45ee0b-0610-40a4-8ea2-2879ace5e6c7

# A Multiscale Whole-Brain Simulator

QUILT (Quick Unified Integration of muLTiscale neural networks) is a neural simulation framework combining **neural mass models** and **spiking neural networks** to simulate whole-brain dynamics across multiple levels of description.

The goal of this project is to bridge large-scale brain activity and fine-grained neuronal dynamics within a single, coherent simulation environment.

---

## Overview

This simulator integrates:
- **Neural Mass Models (NMMs)** for large-scale, population-level brain dynamics  
- **Spiking Neural Networks (SNNs)** for detailed, neuron-level activity  
- A coupling mechanism that allows information to flow **across scales**

---

## Motivation

Most brain simulators focus on a single scale:
- either **macroscopic** (neural mass / mean-field models)
- or **microscopic** (spiking neurons)

This project explores what happens when you **don’t choose** — and instead let multiple scales interact dynamically.

The simulator is intended as a research and experimentation tool, not a finalized biological model.

---

## Current Status

⚠️ **Work in progress**

The simulator is under active development.  
APIs, model implementations, and parameter choices may change.

Expect rough edges. That’s the point.


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
