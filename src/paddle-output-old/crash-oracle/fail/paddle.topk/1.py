import paddle
arg_1_tensor = paddle.randint(-32768,256,[2, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = -46
res = paddle.topk(arg_1,k=arg_2,axis=arg_3,)
