import paddle

x = paddle.to_tensor([1, 2, 3])
y = paddle.diagflat(x)
print(y.numpy())
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

y = paddle.diagflat(x, offset=1)
print(y.numpy())
# [[0 1 0 0]
#  [0 0 2 0]
#  [0 0 0 3]
#  [0 0 0 0]]

y = paddle.diagflat(x, offset=-1)
print(y.numpy())
# [[0 0 0 0]
#  [1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]]