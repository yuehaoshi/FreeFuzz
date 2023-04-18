import paddle

dense_x = paddle.to_tensor([-2, 0, 1], dtype='float32')
sparse_x = dense_x.to_sparse_coo(1)
out = paddle.sparse.log1p(sparse_x)