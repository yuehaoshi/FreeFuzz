import paddle
arg_1_tensor = paddle.randint(-2048,32,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 13.0
res = paddle.cast(arg_1,arg_2,)
