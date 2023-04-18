import paddle

dense_x = paddle.to_tensor([-180, 0, 180])
sparse_x = dense_x.to_sparse_coo(1)
out = paddle.sparse.deg2rad(sparse_x)