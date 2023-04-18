import paddle
arg_class = paddle.nn.Tanhshrink()
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = arg_class(*arg_1)
