import paddle
arg_1_tensor = paddle.randint(0,2,[2, 2])
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.all(arg_1,axis=arg_2,)
