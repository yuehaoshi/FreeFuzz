import paddle

indices = [[0, 0, 1], [1, 1, 2]]
values = [1.0, 2.0, 3.0]
sp_x = paddle.sparse.sparse_coo_tensor(indices, values)
sp_x = paddle.sparse.coalesce(sp_x)
print(sp_x.indices())
#[[0, 1], [1, 2]]
print(sp_x.values())
#[3.0, 3.0]