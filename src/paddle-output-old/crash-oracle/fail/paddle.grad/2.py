import paddle
arg_1_0_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0_tensor = paddle.rand([1], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3 = None
res = paddle.grad(outputs=arg_1,inputs=arg_2,grad_outputs=arg_3,)
