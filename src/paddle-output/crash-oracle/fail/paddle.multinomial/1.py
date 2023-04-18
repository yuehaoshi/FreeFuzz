import paddle
arg_1_tensor = paddle.rand([52, 4, 1], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
res = paddle.multinomial(arg_1,num_samples=arg_2,)
