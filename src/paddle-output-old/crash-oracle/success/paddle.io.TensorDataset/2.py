import paddle
arg_1_0_tensor = paddle.rand([2, 3, 4], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8,512,[2, 1], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = paddle.io.TensorDataset(arg_1,)
