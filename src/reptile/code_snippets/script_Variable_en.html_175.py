import paddle

x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1, 2])
output = paddle.unique_consecutive(x) #
np_output = output.numpy() # [1 2 3 1 2]
_, inverse, counts = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
np_inverse = inverse.numpy() # [0 0 1 1 2 3 3 4]
np_counts = inverse.numpy() # [2 2 1 2 1]

x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
output = paddle.unique_consecutive(x, axis=0) #
np_output = output.numpy() # [2 1 3 0 1 2 1 3 2 1 3]

x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
output = paddle.unique_consecutive(x, axis=0) #
np_output = output.numpy()
# [[2 1 3]
#  [3 0 1]
#  [2 1 3]]