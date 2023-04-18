import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,256,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.divide(arg_1,arg_2,)
