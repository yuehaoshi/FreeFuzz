import paddle

dense_x = paddle.to_tensor([[-2., 0.], [1., 2.]])
sparse_x = dense_x.to_sparse_coo(1)
out = paddle.sparse.transpose(sparse_x, [1, 0])