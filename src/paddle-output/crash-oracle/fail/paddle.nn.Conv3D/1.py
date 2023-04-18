import paddle
arg_1 = 4
arg_2 = 1
arg_3_0 = 3
arg_3_1 = 3
arg_3_2 = 3
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
arg_class = paddle.nn.Conv3D(arg_1,arg_2,arg_3,)
arg_4_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
res = arg_class(*arg_4)
