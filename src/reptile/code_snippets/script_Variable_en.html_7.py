import paddle.fluid as fluid
import numpy as np

x = np.ones([2, 2], np.float32)
with fluid.dygraph.guard():
    inputs2 = []
    for _ in range(10):
        tmp = fluid.dygraph.base.to_variable(x)
        tmp.stop_gradient=False
        inputs2.append(tmp)
    ret2 = fluid.layers.sums(inputs2)
    loss2 = fluid.layers.reduce_sum(ret2)
    loss2.backward()
    print(loss2.gradient())
    loss2.clear_gradient()
    print("After clear {}".format(loss2.gradient()))