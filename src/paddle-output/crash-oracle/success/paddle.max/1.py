import paddle
arg_1_tensor = paddle.randint(-8192,4,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.max(arg_1,axis=arg_2,)
