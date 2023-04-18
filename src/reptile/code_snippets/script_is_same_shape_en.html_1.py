import paddle

x = paddle.rand([2, 3, 8])
y = paddle.rand([2, 3, 8])
y = y.to_sparse_csr()
z = paddle.rand([2, 5])

paddle.sparse.is_same_shape(x, y)
# True
paddle.sparse.is_same_shape(x, z)
# False