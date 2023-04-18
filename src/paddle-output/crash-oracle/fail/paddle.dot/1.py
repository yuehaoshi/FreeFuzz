import paddle
arg_1_tensor = paddle.rand([10, 26], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,8192,[4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.dot(arg_1,arg_2,)
