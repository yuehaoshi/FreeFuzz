import paddle
arg_1_tensor = paddle.rand([4, 64], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.argmax(arg_1,axis=arg_2,)
