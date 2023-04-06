import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    new_variable = fluid.dygraph.to_variable(np.arange(10))