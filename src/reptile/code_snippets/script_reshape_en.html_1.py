import paddle

x_shape = [6, 2, 3]
new_shape = [1, 0, 2, -1, 3]
format = "coo"

dense_x = paddle.randint(-100, 100, x_shape) * paddle.randint(0, 2, x_shape)

if format == "coo":
    sp_x = dense_x.to_sparse_coo(len(x_shape))
else:
    sp_x = dense_x.to_sparse_csr()
sp_out = paddle.sparse.reshape(sp_x, new_shape)

print(sp_out)
# the shape of sp_out is [1, 2, 2, 3, 3]