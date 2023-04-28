import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "int8"
res = paddle.Tensor.astype(arg_1,arg_2,)
