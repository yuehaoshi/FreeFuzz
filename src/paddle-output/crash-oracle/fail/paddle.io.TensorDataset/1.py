import paddle
arg_1_0_tensor = paddle.rand([2, 3, 4], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8192,256,[2, 1], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_class = paddle.io.TensorDataset(arg_1,)
arg_2 = "max"
res = arg_class(*arg_2)
