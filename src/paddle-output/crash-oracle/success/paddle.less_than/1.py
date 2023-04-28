import paddle
arg_1_tensor = paddle.randint(-256,32768,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,8,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.less_than(x=arg_1,y=arg_2,)
