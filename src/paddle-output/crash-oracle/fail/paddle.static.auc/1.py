import paddle
arg_1_tensor = paddle.rand([-1, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,512,[-1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.static.auc(input=arg_1,label=arg_2,)
