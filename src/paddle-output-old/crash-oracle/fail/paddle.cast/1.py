import paddle
arg_1_tensor = paddle.randint(-32,32768,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "paddleVarType"
res = paddle.cast(arg_1,arg_2,)
