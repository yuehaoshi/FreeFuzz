import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = False
arg_4 = "equal_nan"
res = paddle.sum(arg_1,arg_2,keepdim=arg_3,name=arg_4,)
