import paddle
arg_1 = "element_wise_add"
arg_2_0_tensor = paddle.randint(-32,256,[2, 3], dtype=paddle.int32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-16,16,[2, 3], dtype=paddle.int32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
arg_3_tensor = paddle.randint(-1024,512,[3, 1], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
res = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
