import paddle
arg_1_tensor = paddle.randint(-256,512,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.argmax(arg_1,)