import paddle
arg_1_tensor = paddle.randint(-32768,16384,[32, 32], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "int64"
res = paddle.ones_like(arg_1,dtype=arg_2,)
