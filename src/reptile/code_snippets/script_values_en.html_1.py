import paddle
from paddle.fluid.framework import _test_eager_guard
with _test_eager_guard():
    indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
    values = [1, 2, 3, 4, 5]
    dense_shape = [3, 4]
    sparse_x = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(indices, dtype='int32'), paddle.to_tensor(values, dtype='float32'), shape=dense_shape)
    print(sparse_x.values())
    #[1, 2, 3, 4, 5]