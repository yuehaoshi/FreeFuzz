import paddle
arg_1_tensor = paddle.randint(-256,4096,[], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "int32"
res = paddle.arange(arg_1,dtype=arg_2,)
