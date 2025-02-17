# Calci

So far, implements a fast Lennard-Jones

## Installation

To install, clone the repository and run `pip install`:
```bash
pip install .
```

*NOTE:* The `pip install` command will try to compile the C++ source code and build a binary wheel. 
Hence, this step depends on your system being equipped with a working `C++` compiler and the necessary build tools. 
We recommend to obtain them from the shipped conda `environment.yml` file. 
See below:

## Obtaining build dependencies
We use micromamba here, but `conda` or `mamba` will work just as well.
First create the environment, then activate it.

```bash
micromamba create -f environment.yml # only once
micromamba activate calcienv # every time 
```
After you have activated the environment it should be possible to install fixi with `pip install .`

## Advanced building and running the tests

We use `meson` as the build system: 

```bash
meson setup build
meson compile -C build
```

To run the `C++` unit tests:
```bash
meson test -C build
```

To run the tests in `Python` (which are in `pytest`): 

```bash
pytest -v
```