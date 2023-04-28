import paddle
arg_1_tensor = paddle.randint(-16384,64,[-1, 5], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.assign(arg_1,)
