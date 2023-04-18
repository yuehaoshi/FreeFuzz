import paddle
arg_1 = -60
arg_2 = 4
arg_3 = 1000
arg_class = paddle.nn.Bilinear(in1_features=arg_1,in2_features=arg_2,out_features=arg_3,)
arg_4 = None
res = arg_class(*arg_4)
