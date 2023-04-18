import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "builtinsset"
res = paddle.static.append_backward(loss=arg_1,no_grad_set=arg_2,)
