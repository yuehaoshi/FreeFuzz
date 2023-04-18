import paddle
arg_1_tensor = paddle.randint(-256,4096,[0, 3, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,16,[1, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.gather_nd(arg_1,arg_2,)
