import paddle
arg_1_tensor = paddle.randint(-512,4,[2, 4, 3], dtype=paddle.int16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-1,8192,[2], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
arg_4 = False
res = paddle.text.viterbi_decode(arg_1,arg_2,arg_3,arg_4,)
