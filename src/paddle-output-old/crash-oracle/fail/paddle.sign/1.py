import paddle
arg_1_tensor = paddle.randint(-2048,16,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.sign(x=arg_1,)
