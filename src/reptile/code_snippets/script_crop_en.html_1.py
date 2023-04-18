import paddle
x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# x.shape = [3, 3]
# x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# shape can be a 1-D Tensor or list or tuple.
shape = paddle.to_tensor([2, 2], dtype='int32')
# shape = [2, 2]
# shape = (2, 2)
out = paddle.crop(x, shape)
# out.shape = [2, 2]
# out = [[1,2], [4,5]]

# offsets can be a 1-D Tensor or list or tuple.
offsets = paddle.to_tensor([0, 1], dtype='int32')
# offsets = [1, 0]
# offsets = (1, 1)
out = paddle.crop(x, shape, offsets)
# out.shape = [2, 2]
# if offsets = [0, 0], out = [[1,2], [4,5]]
# if offsets = [0, 1], out = [[2,3], [5,6]]
# if offsets = [1, 0], out = [[4,5], [7,8]]
# if offsets = [1, 1], out = [[5,6], [8,9]]