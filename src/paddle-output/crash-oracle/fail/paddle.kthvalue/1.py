import paddle
arg_1_tensor = paddle.rand([2, 3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -29
arg_3 = 16
res = paddle.kthvalue(arg_1,arg_2,arg_3,)
