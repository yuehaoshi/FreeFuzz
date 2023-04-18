import paddle

indices = [[0, 1, 2], [1, 2, 0]]
values = [1.0, 2.0, 3.0]
dense_shape = [3, 3]
coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
# print(coo)
# Tensor(shape=[2, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
#       indices=[[0, 1, 2],
#                [1, 2, 0]],
#       values=[1., 2., 3.])