import paddle
arg_1_tensor = paddle.rand([5, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,64,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.unsqueeze(arg_1,axis=arg_2,)
