# Instrumentation 

This folder contains code to perform instrumentation on Pytorch in order to collect dynamic execution information.

We hook the invocation of `5` PaddlePaddle APIs in total, and the API names are listed in `paddle.*` files.

## Usage:

(1) Copy the files (except `__init__.py`) under this `instrumentation` folder to the root directory where Pytorch is installed. You may want to obtain the path by running the following commands:
```
pip show paddlepaddle
```
And it should return something similar to `.../lib64/python3.6/site-packages/paddle`.

(2) Append the lines from the file `__init__.py` in this directory to the end of the `__init__.py` file in the root directory of installed pytorch, which should be similar to `.../lib64/python3.6/site-packages/paddle/__init__.py`

(3) Configure your MongoDB in the file `write_tools.py` and then run the code where Paddlepaddle APIs are invoked. The traced dynamic execution information for each API invocation will be added to the MongoDB.
