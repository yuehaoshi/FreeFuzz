import paddle
import numpy as np

inp_np = np.ones([5, 2, 3, 4]).astype('float32')
inp_np = paddle.to_tensor(inp_np)
flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
flatten_res = flatten(inp_np)