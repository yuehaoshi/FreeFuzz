import paddle
arg_1_tensor = paddle.randint(-256,2,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_3 = True
res = paddle.median(arg_1,axis=arg_2,keepdim=arg_3,)
