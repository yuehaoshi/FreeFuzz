# required: gpu
import paddle
from paddle.fluid.framework import _test_eager_guard
paddle.seed(100)

# csr @ dense -> dense
with _test_eager_guard():
    crows = [0, 2, 3, 5]
    cols = [1, 3, 2, 0, 1]
    values = [1., 2., 3., 4., 5.]
    dense_shape = [3, 4]
    csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)
    # Tensor(shape=[3, 4], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
    #        crows=[0, 2, 3, 5],
    #        cols=[1, 3, 2, 0, 1],
    #        values=[1., 2., 3., 4., 5.])
    vec = paddle.randn([4])

    out = paddle.sparse.mv(csr, vec)
    # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
    #        [-3.85499096, -2.42975140, -1.75087738])