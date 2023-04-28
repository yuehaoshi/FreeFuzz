import paddle
arg_1_tensor = paddle.rand([2, 3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = "int64"
res = paddle.prod(arg_1,arg_2,dtype=arg_3,)
