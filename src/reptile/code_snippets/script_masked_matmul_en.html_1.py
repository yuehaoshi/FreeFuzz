# required: gpu
import paddle
paddle.seed(100)

# dense @ dense * csr_mask -> csr
crows = [0, 2, 3, 5]
cols = [1, 3, 2, 0, 1]
values = [1., 2., 3., 4., 5.]
dense_shape = [3, 4]
mask = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
# Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
#       crows=[0, 2, 3, 5],
#       cols=[1, 3, 2, 0, 1],
#       values=[1., 2., 3., 4., 5.])

x = paddle.rand([3, 5])
y = paddle.rand([5, 4])

out = paddle.sparse.masked_matmul(x, y, mask)
# Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
#        crows=[0, 2, 3, 5],
#        cols=[1, 3, 2, 0, 1],
#        values=[0.98986477, 0.97800624, 1.14591956, 0.68561077, 0.94714981])