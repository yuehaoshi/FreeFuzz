import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3.5
arg_3 = 5.0
res = paddle.clip(arg_1,min=arg_2,max=arg_3,)
