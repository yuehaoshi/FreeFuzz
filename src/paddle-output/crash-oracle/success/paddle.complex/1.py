import paddle
arg_1_tensor = paddle.rand([8, 257, 376], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([8, 257, 376], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.complex(arg_1,arg_2,)