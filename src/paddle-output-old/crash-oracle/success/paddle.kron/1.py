import paddle
arg_1_tensor = paddle.randint(-8192,2048,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,1,[3, 3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.kron(arg_1,arg_2,)