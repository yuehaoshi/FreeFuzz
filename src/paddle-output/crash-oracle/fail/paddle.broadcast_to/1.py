import paddle
arg_1_tensor = paddle.randint(-8192,16,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -23
arg_2 = [arg_2_0,]
res = paddle.broadcast_to(arg_1,arg_2,)
