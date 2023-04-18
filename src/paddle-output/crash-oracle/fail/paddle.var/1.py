import paddle
arg_1_tensor = paddle.randint(-512,1024,[4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.var(arg_1,axis=arg_2,)
