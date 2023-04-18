import paddle
arg_1_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.75
res = paddle.pow(arg_1,arg_2,)
