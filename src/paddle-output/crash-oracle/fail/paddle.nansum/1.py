import paddle
arg_1_tensor = paddle.rand([2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_3 = True
res = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,)
