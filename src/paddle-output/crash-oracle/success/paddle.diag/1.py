import paddle
arg_1_tensor = paddle.randint(-512,512,[2, 2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = 38
res = paddle.diag(arg_1,padding_value=arg_2,)
