import paddle
arg_1_tensor = paddle.rand([3, 10, 5, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = 1
arg_4 = -1
res = paddle.trace(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
