import paddle
arg_1_tensor = paddle.randint(-128,16,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 63.0
res = paddle.argmin(arg_1,axis=arg_2,)
