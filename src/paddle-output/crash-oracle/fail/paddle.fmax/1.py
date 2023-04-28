import paddle
arg_1_tensor = paddle.rand([100, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,512,[2, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fmax(arg_1,arg_2,)
