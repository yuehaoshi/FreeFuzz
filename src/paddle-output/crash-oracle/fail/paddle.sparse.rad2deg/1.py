import paddle
arg_1_tensor = paddle.randint(-2,64,[], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.sparse.rad2deg(arg_1,)
