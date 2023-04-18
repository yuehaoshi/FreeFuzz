import paddle

x = paddle.to_tensor([[1, 2], [7, 8]])
y = paddle.to_tensor([[3, 4], [5, 6]])
res = paddle.minimum(x, y)
print(res)
# Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[1, 2],
#         [5, 6]])

x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
y = paddle.to_tensor([3, 0, 4])
res = paddle.minimum(x, y)
print(res)
# Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[[1, 0, 3],
#          [1, 0, 3]]])

x = paddle.to_tensor([2, 3, 5], dtype='float32')
y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
res = paddle.minimum(x, y)
print(res)
# Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [1. , nan, nan])

x = paddle.to_tensor([5, 3, float("inf")], dtype='float64')
y = paddle.to_tensor([1, -float("inf"), 5], dtype='float64')
res = paddle.minimum(x, y)
print(res)
# Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [ 1.  , -inf.,  5.  ])