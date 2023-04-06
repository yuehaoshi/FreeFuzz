import paddle

x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
unique = paddle.unique(x)
np_unique = unique.numpy() # [1 2 3 5]
_, indices, inverse, counts = paddle.unique(x, return_index=True, return_inverse=True, return_counts=True)
np_indices = indices.numpy() # [3 0 1 4]
np_inverse = inverse.numpy() # [1 2 2 0 3 2]
np_counts = counts.numpy() # [1 1 3 1]

x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
unique = paddle.unique(x)
np_unique = unique.numpy() # [0 1 2 3]

unique = paddle.unique(x, axis=0)
np_unique = unique.numpy()
# [[2 1 3]
#  [3 0 1]]