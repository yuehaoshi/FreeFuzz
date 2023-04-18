import paddle
arg_1_tensor = paddle.randint(-8192,256,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,1024,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.equal(arg_1,arg_2,)
