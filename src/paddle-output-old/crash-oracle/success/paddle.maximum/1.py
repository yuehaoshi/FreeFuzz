import paddle
arg_1_tensor = paddle.rand([2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.maximum(arg_1,arg_2,)
