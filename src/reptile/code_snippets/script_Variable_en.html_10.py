import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    value0 = np.arange(26).reshape(2, 13).astype("float32")
    value1 = np.arange(6).reshape(2, 3).astype("float32")
    value2 = np.arange(10).reshape(2, 5).astype("float32")
    linear = fluid.Linear(13, 5, dtype="float32")
    linear2 = fluid.Linear(3, 3, dtype="float32")
    a = fluid.dygraph.to_variable(value0)
    b = fluid.dygraph.to_variable(value1)
    c = fluid.dygraph.to_variable(value2)
    out1 = linear(a)
    out2 = linear2(b)
    out1.stop_gradient = True
    out = fluid.layers.concat(input=[out1, out2, c], axis=1)
    out.backward()

    assert linear.weight.gradient() is None
    assert (out1.gradient() == 0).all()