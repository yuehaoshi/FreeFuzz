import paddle

dense_x = paddle.to_tensor([-2, 0, 3], dtype='float32')
sparse_x = dense_x.to_sparse_coo(1)
out = paddle.sparse.pow(sparse_x, 2)