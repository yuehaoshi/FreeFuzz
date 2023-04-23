# Instrumentation 

This folder contains code to perform instrumentation on Pytorch in order to collect dynamic execution information.

We hook the invocation of `827` PaddlePaddle APIs in total, and the API names are listed in `paddle.txt` file.
## Preparation

You can collect API data by following the `README` file in `FreeFuzz/src/reptile`, which contains the reptile code to get official documentation API lists and execution

## Usage:

(1) Copy the files (except `__init__.py`) under this `instrumentation/paddlepaddle` folder to the root directory where Pytorch is installed. You may want to obtain the path by running the following commands:
```
pip show paddlepaddle
```
And it should return something similar to `.../lib64/python3.6/site-packages/paddle`.

(2) Append the lines from the file `__init__.py` in this directory to the end of the `__init__.py` file in the root directory of installed pytorch, which should be similar to `.../lib64/python3.6/site-packages/paddle/__init__.py`

(3) Configure your MongoDB in the file `write_tools.py` and then run the code where Paddlepaddle APIs are invoked  
For example, to run all official documentation API execution examples, you should run all `.py` files in `/src/reptile/code_snippets`, where all code snippets reptiled from PaddlePaddle 2.4 version are stored). The traced dynamic execution information for each API invocation will be added to the MongoDB.  
You can run all `.py` files in `/code_snippets` by running `cd code_snippets` and then `find . -type f -name "*.py" | xargs -I{} -n 1 sh -c 'echo "Running {}..."; python "{}"; echo "Finished {}"'`  
(or `find . -type f -name "*.py" | sort | xargs -I{} -n 1 sh -c 'echo "Running {}..."; python "{}"; echo "Finished {}"' > output.txt` if you want to run files by file name order and save terminal output)
Two code snippets `script_Model_en.html_11.py` and `script_Model_en.html_12.py` were deleted because they will crush the automatic running process.
