import paddle
arg_1_tensor = paddle.rand([-1, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,1,[-1, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 5
res = paddle.static.accuracy(input=arg_1,label=arg_2,k=arg_3,)
