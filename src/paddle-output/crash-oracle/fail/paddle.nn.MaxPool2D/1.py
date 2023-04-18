import paddle
arg_1 = 2
arg_2 = 2
arg_class = paddle.nn.MaxPool2D(arg_1,arg_2,)
arg_3_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = arg_class(*arg_3)
