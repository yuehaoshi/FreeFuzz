import paddle.fluid as fluid
import numpy as np

x = np.ones([2, 2], np.float32)
with fluid.dygraph.guard():
    original_variable = fluid.dygraph.to_variable(x)
    print("original var's dtype is: {}, numpy dtype is {}".format(original_variable.dtype, original_variable.numpy().dtype))
    new_variable = original_variable.astype('int64')
    print("new var's dtype is: {}, numpy dtype is {}".format(new_variable.dtype, new_variable.numpy().dtype))