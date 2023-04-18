import paddle
arg_1_tensor = paddle.randint(-128,256,[4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -16
res = paddle.topk(arg_1,k=arg_2,)
