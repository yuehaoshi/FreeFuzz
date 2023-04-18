import paddle
arg_1_tensor = paddle.rand([2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 0
arg_2_2 = 2
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = paddle.transpose(arg_1,perm=arg_2,)
