import paddle
arg_1_tensor = paddle.randint(-1024,4,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,2048,[3, 5], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = "wrap"
res = paddle.take(arg_1,arg_2,mode=arg_3,)
