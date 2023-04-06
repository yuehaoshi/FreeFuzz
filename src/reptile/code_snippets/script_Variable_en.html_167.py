import paddle

data = paddle.to_tensor([1, 2, 3], dtype='int32')
out = paddle.tile(data, repeat_times=[2, 1])
np_out = out.numpy()
# [[1, 2, 3], [1, 2, 3]]

out = paddle.tile(data, repeat_times=[2, 2])
np_out = out.numpy()
# [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]

repeat_times = paddle.to_tensor([2, 1], dtype='int32')
out = paddle.tile(data, repeat_times=repeat_times)
np_out = out.numpy()
# [[1, 2, 3], [1, 2, 3]]