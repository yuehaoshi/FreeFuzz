import paddle
arg_1_tensor = paddle.randint(-32,4096,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "float64"
res = paddle.zeros_like(arg_1,dtype=arg_2,)
