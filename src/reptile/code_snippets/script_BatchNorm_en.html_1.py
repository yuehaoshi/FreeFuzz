import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import numpy as np

x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
with fluid.dygraph.guard():
    x = to_variable(x)
    batch_norm = fluid.BatchNorm(10)
    hidden1 = batch_norm(x)