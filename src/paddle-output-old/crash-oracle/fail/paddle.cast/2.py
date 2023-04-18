import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = "float32"
res = paddle.cast(arg_1,arg_2,)
