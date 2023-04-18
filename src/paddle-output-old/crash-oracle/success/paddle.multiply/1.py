import paddle
arg_1_tensor = paddle.randint(-64,64,[1, 2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,256,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.multiply(arg_1,arg_2,)
