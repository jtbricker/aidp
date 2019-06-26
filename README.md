# AIDP Tool
[![Build Status](https://travis-ci.org/jtbricker/aidp.svg?branch=master)](https://travis-ci.org/jtbricker/aidp)
[![Coverage Status](https://coveralls.io/repos/github/jtbricker/aidp/badge.svg?branch=master)](https://coveralls.io/github/jtbricker/aidp?branch=master)

A script for getting predictions from the AIDP model

## Installation
Running 
``` bash
pip install -e .
```
in the aidp directory should make the `aidp` command available to use anywhere.

## Usage

``` bash
aidp predict input_file.xlsx output_file.xlsx
```

Run

``` bash
aidp --help
``` 
for a full list of options and argument descriptions.


## TODO:

* Add Tests where missing
* Comment the code where necessary
* Fully implement training module
* Handle Deprecation / Data Casting Warnings that are printed to the console.
* Update README with additinal installation / usage instructions