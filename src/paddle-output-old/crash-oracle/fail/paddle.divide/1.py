import paddle
arg_1_tensor = paddle.rand([2, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
res = paddle.divide(arg_1,arg_2,name=arg_3,)
