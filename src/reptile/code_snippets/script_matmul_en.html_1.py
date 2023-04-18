# required: gpu
import paddle

# csr @ dense -> dense
crows = [0, 1, 2, 3]
cols = [1, 2, 0]
values = [1., 2., 3.]
csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, [3, 3])
# Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
#        crows=[0, 1, 2, 3],
#        cols=[1, 2, 0],
#        values=[1., 2., 3.])
dense = paddle.ones([3, 2])
out = paddle.sparse.matmul(csr, dense)
# Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[1., 1.],
#         [2., 2.],
#         [3., 3.]])

# coo @ dense -> dense
indices = [[0, 1, 2], [1, 2, 0]]
values = [1., 2., 3.]
coo = paddle.sparse.sparse_coo_tensor(indices, values, [3, 3])
# Tensor(shape=[3, 3], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
#        indices=[[0, 1, 2],
#                 [1, 2, 0]],
#        values=[1., 2., 3.])
dense = paddle.ones([3, 2])
out = paddle.sparse.matmul(coo, dense)
# Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[1., 1.],
#         [2., 2.],
#         [3., 3.]])