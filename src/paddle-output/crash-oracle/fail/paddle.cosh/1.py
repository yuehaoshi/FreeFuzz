import paddle
arg_1_tensor = paddle.randint(-128,256,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.cosh(arg_1,)
