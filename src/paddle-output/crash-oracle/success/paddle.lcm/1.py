import paddle
arg_1_tensor = paddle.randint(-64,1,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,1024,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.lcm(arg_1,arg_2,)
