import paddle
arg_1_tensor = paddle.randint(-8,2,[2, 2], dtype=paddle.int16)
arg_1 = arg_1_tensor.clone()
arg_2 = 6
res = paddle.any(arg_1,axis=arg_2,)
