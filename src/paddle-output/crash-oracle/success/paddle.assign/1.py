import paddle
arg_1_tensor = paddle.randint(-32,4096,[2, 2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.assign(arg_1,)
