import paddle
arg_1_tensor = paddle.randint(-8192,512,[1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,4,[1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.equal(arg_1,arg_2,)
