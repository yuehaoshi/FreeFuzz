import paddle
arg_1_tensor = paddle.randint(-8,1024,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.multinomial(arg_1,)
