import os
import paddle

paddle.jit.set_verbosity(1)
# The verbosity level is now 1

os.environ['TRANSLATOR_VERBOSITY'] = '3'
# The verbosity level is now 3, but it has no effect because it has a lower priority than `set_verbosity`