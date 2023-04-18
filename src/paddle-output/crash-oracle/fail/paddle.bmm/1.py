import paddle
arg_1_0_tensor = paddle.randint(-512,16,[2, 3], dtype=paddle.int32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-32,1,[2, 3], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2_tensor = paddle.rand([2, 3, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.bmm(arg_1,arg_2,)
