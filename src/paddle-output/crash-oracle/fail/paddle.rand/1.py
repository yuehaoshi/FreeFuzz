import paddle
arg_1_tensor = paddle.randint(-64,4,[2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = "paddleVarType"
res = paddle.rand(arg_1,dtype=arg_2,)
