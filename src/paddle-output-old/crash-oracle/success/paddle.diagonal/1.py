import paddle
arg_1_tensor = paddle.rand([2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 28
arg_3 = 0
arg_4 = 1
res = paddle.diagonal(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
