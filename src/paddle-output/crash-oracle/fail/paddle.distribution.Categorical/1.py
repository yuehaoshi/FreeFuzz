import paddle
arg_1_tensor = paddle.rand([6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_class = paddle.distribution.Categorical(arg_1,)
arg_2 = None
res = arg_class(*arg_2)
