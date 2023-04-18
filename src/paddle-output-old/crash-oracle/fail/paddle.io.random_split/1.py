import paddle
arg_1 = True
arg_2_tensor = paddle.randint(-2048,16384,[2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.io.random_split(arg_1,arg_2,)
