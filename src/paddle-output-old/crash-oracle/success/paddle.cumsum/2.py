import paddle
arg_1_tensor = paddle.randint(-4096,16,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "float64"
res = paddle.cumsum(arg_1,dtype=arg_2,)
