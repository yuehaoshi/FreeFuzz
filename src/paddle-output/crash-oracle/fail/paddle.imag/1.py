import paddle
arg_1_tensor = paddle.randint(0,2,[2, 3], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
res = paddle.imag(arg_1,)