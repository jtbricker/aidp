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
aidp --help
``` 
for a full list of options and argument descriptions.


### Train
Pass `train` as the second parameter to `aidp` to train a new model based on the data in the excel sheet passed as the third parameter. E.g,

``` bash
aidp train /path/to/input_file.xlsx [--model_key='default']
```

The input excel sheet must adhere pretty closely to the format of the sample excel sheet in `./tests/resources/text.xlsx`.

You can optionally provide a model key (using  `--model_key=<key>`) when making a call to train the models.  This will save all models generated during this session to `/resources/models/<key>/` folder.  If no model key is provided, a timestamp will be used. 



### Predict
Pass `predict` as the second parameter to `aidp` to make predictions using the model specified by the `model_key` parameter based on the data in the excel sheet passed as the third parameter. E.g,

``` bash
aidp predict input_file.xlsx [--model_key='default']
```

Again,the input excel sheet must adhere pretty closely to the format of the sample excel sheet in `./tests/resources/text.xlsx`. However, no GroupIds are required for predictions.

You can optionally provide a model key (using  `--model_key=<key>`) when making a call to predict.  This will use the models saved in `/resources/models/<key>/` folder for predictions.  If no model key is provided, `default` is used. 
