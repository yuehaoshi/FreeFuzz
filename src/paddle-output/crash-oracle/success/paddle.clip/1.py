import paddle
arg_1_tensor = paddle.rand([3, 55, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
arg_3 = 9
res = paddle.clip(arg_1,arg_2,arg_3,)
