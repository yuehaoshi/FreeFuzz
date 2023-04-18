import paddle
arg_1 = "slot4"
arg_2_0 = None
arg_2_1 = 40
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "float32"
arg_4 = 1
res = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,lod_level=arg_4,)
