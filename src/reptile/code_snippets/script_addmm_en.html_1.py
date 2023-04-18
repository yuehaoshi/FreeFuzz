# required: gpu
import paddle

# dense + csr @ dense -> dense
input = paddle.rand([3, 2])
crows = [0, 1, 2, 3]
cols = [1, 2, 0]
values = [1., 2., 3.]
x = paddle.sparse.sparse_csr_tensor(crows, cols, values, [3, 3])
y = paddle.rand([3, 2])
out = paddle.sparse.addmm(input, x, y, 3.0, 2.0)

# dense + coo @ dense -> dense
input = paddle.rand([3, 2])
indices = [[0, 1, 2], [1, 2, 0]]
values = [1., 2., 3.]
x = paddle.sparse.sparse_coo_tensor(indices, values, [3, 3])
y = paddle.rand([3, 2])
out = paddle.sparse.addmm(input, x, y, 3.0, 2.0)