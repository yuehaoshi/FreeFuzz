import paddle
arg_1_tensor = paddle.randint(-4,2048,[2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = "int64"
res = paddle.ones(arg_1,dtype=arg_2,)
