import paddle
arg_1_tensor = paddle.randint(0,2,[2, 2], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,32,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.unsqueeze(arg_1,axis=arg_2,)
