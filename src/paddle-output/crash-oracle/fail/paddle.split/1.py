import paddle
arg_1_tensor = paddle.rand([16, 96], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_3 = -54
res = paddle.split(arg_1,num_or_sections=arg_2,axis=arg_3,)
