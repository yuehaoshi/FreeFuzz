import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import Linear
import numpy as np

data = np.ones([3, 1024], dtype='float32')
with fluid.dygraph.guard():
    linear = fluid.dygraph.Linear(1024, 4)
    t = to_variable(data)
    linear(t)  # call with default weight
    custom_weight = np.random.randn(1024, 4).astype("float32")
    linear.weight.set_value(custom_weight)  # change existing weight
    out = linear(t)  # call with different weight