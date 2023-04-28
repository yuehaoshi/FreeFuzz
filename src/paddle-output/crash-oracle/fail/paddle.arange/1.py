import paddle
arg_1_tensor = paddle.randint(-64,1,[1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,64,[1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = -93.0
res = paddle.arange(arg_1,arg_2,dtype=arg_3,)
