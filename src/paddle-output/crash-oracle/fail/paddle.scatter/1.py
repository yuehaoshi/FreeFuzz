import paddle
arg_1_tensor = paddle.randint(-4096,256,[60000], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,512,[60000], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16,8,[60000], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
res = paddle.scatter(arg_1,arg_2,arg_3,)
