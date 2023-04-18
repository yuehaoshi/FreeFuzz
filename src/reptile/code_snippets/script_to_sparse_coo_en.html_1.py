import paddle
from paddle.fluid.framework import _test_eager_guard
with _test_eager_guard():
    dense_x = [[0, 1, 0, 2], [0, 0, 3, 4]]
    dense_x = paddle.to_tensor(dense_x, dtype='float32')
    sparse_x = dense_x.to_sparse_coo(sparse_dim=2)
    #indices=[[0, 0, 1, 1],
    #         [1, 3, 2, 3]],
    #values=[1., 2., 3., 4.]