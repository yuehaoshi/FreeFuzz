import paddle
arg_1_tensor = paddle.rand([1, 3, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,2048,[4], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.crop(arg_1,shape=arg_2,)
