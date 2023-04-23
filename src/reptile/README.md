# Usage and Introduction for each file:

To get Paddle official API names, links and execution example, you can follow the following steps:

1. Run `get_API_list.py` file, this will generate `API_lists.txt` and `API_links.txt`, which contains API names and links respectively
2. Run `get_API_execution.py` file, this will generate a folder `code_snippets` in your current folder, containing lots of `.py` files which are the execution examples got from Paddle API website.
3. Run `get_API_definition.py` file, this will generate a file `API_def.txt` containing all API definitions defined in Paddle website
4. Run `API_cleaner.py` file, this will clean the ugly-formatted API definition and save output into a file `API_def_mod`. The code removes spaces before or after hyphens, replaces multiple spaces with a single space, removes the prefix "class " if it appears at the beginning of a line, removes everything after the string "[source]", and skips lines that do not end with a closing parenthesis.

Delete `paddle.Tensor.Overview` from `API_lists` because it is not an API, a wrong reptile result.
Next, copy all contents from `API_def_mod.txt` to `FreeFuzz/data/paddle_APIdef.txt`, and copy all contents from `API_lists.txt` to `FreeFuzz/instrumentation/paddlepaddle/paddle.txt`


