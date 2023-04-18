import paddle.fluid as fluid
import numpy as np

# example1: return ndarray
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

# example2: return tuple of ndarray
with fluid.dygraph.guard():
    embedding = fluid.dygraph.Embedding(
        size=[20, 32],
        param_attr='emb.w',
        is_sparse=True)
    x_data = np.arange(12).reshape(4, 3).astype('int64')
    x_data = x_data.reshape((-1, 3, 1))
    x = fluid.dygraph.base.to_variable(x_data)
    out = embedding(x)
    out.backward()
    print(embedding.weight.gradient())