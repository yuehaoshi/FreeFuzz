import paddle
arg_1_tensor = paddle.rand([4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,16,[10], dtype=paddle.int16)
arg_2 = arg_2_tensor.clone()
res = paddle.dot(arg_1,arg_2,)
