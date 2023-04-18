import paddle

crows = [0, 2, 3, 5]
cols = [1, 3, 2, 0, 1]
values = [1, 2, 3, 4, 5]
dense_shape = [3, 4]
csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
# print(csr)
# Tensor(shape=[3, 4], dtype=paddle.int64, place=Place(gpu:0), stop_gradient=True,
#       crows=[0, 2, 3, 5],
#       cols=[1, 3, 2, 0, 1],
#       values=[1, 2, 3, 4, 5])