import paddle
arg_1_tensor = paddle.rand([6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -20.99999
arg_4 = 1e-08
arg_5 = False
arg_6 = "ignore_nan"
res = paddle.allclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
