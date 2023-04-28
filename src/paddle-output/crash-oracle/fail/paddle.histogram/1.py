import paddle
arg_1_0_tensor = paddle.rand([1, 1, 28, 28], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([1, 400], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 4
arg_3 = 0
arg_4 = 3
res = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
