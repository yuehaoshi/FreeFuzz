import paddle
arg_1_tensor = paddle.rand([4, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 4
res = paddle.topk(x=arg_1,k=arg_2,)
