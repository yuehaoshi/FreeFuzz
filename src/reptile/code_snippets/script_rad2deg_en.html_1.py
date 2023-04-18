import paddle

dense_x = paddle.to_tensor([3.142, 0., -3.142])
sparse_x = dense_x.to_sparse_coo(1)
out = paddle.sparse.rad2deg(sparse_x)