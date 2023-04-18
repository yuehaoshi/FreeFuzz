import paddle
arg_1_tensor = paddle.randint(0,2,[4], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([4, 1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([4], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.where(arg_1,arg_2,arg_3,)
