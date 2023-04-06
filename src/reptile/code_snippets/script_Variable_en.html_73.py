import paddle

data_x = paddle.to_tensor([1, 2, 3], 'int32')
data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
out = paddle.expand_as(data_x, data_y)
np_out = out.numpy()
# [[1, 2, 3], [1, 2, 3]]