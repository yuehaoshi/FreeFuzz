import paddle
arg_1_0_tensor = paddle.rand([1, 2, 3, 4], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([1, 2, 1, 4], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.rand([1, 1, 3, 1], dtype=paddle.float32)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2_tensor = paddle.randint(-1,16,[-1, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 5
res = paddle.static.accuracy(input=arg_1,label=arg_2,k=arg_3,)
