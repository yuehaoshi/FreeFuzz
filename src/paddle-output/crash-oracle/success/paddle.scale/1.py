import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 50.0
arg_3 = 1.0
res = paddle.scale(arg_1,scale=arg_2,bias=arg_3,)
