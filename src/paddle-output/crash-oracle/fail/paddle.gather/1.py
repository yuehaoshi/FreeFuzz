import paddle
arg_1_tensor = paddle.randint(-256,512,[1, 17], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,64,[5], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.gather(arg_1,arg_2,)
