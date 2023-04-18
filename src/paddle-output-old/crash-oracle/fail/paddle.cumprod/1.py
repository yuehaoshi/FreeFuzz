import paddle
arg_1_tensor = paddle.randint(-128,64,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -46
res = paddle.cumprod(arg_1,dim=arg_2,)
