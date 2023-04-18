import paddle
arg_1_tensor = paddle.randint(-4,256,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.real(arg_1,)
