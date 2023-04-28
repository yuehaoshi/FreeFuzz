import paddle
arg_1_tensor = paddle.rand([5, 12000], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,2048,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.gcd(arg_1,arg_2,)
