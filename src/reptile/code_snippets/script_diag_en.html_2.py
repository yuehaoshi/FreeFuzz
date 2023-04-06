import paddle

paddle.disable_static()
x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
y = paddle.diag(x)
print(y.numpy())
# [1 5]

y = paddle.diag(x, offset=1)
print(y.numpy())
# [2 6]

y = paddle.diag(x, offset=-1)
print(y.numpy())
# [4]