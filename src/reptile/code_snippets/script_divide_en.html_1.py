import paddle

paddle.device.set_device("cpu")

x = paddle.to_tensor([[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]], 'float32')
y = paddle.to_tensor([[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]], 'float32')
sparse_x = x.to_sparse_csr()
sparse_y = y.to_sparse_csr()
sparse_z = paddle.sparse.divide(sparse_x, sparse_y)
print(sparse_z.to_dense())

# [[ nan      , -inf.     ,  nan      , -1.       ],
# [ nan      ,  0.       ,  1.       ,  nan      ],
# [ 2.       , 1.66666663,  0.       ,  0.       ]]