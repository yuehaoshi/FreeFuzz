import paddle
arg_1_tensor = paddle.randint(-32768,32,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.flatten(arg_1,)
