import paddle

paddle.disable_static()
x = paddle.to_tensor([1, 2, 3])
y = paddle.diag(x)
print(y)
# Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[1, 0, 0],
#         [0, 2, 0],
#         [0, 0, 3]])

y = paddle.diag(x, offset=1)
print(y)
# Tensor(shape=[4, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[0, 1, 0, 0],
#         [0, 0, 2, 0],
#         [0, 0, 0, 3],
#         [0, 0, 0, 0]])

y = paddle.diag(x, padding_value=6)
print(y)
# Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[1, 6, 6],
#         [6, 2, 6],
#         [6, 6, 3]])