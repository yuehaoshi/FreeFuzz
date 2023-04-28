import paddle
arg_1_tensor = paddle.randint(-1,128,[1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.acos(arg_1,)
