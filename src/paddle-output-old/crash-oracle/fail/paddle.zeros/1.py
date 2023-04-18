import paddle
arg_1_tensor = paddle.randint(-16384,1,[2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.zeros(shape=arg_1,dtype=arg_2,)
