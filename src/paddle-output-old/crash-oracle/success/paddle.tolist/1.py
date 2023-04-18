import paddle
arg_1_tensor = paddle.randint(-8192,64,[5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.tolist(arg_1,)
