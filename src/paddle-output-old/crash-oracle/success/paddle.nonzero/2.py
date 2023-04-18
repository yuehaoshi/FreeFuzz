import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
res = paddle.nonzero(arg_1,as_tuple=arg_2,)
