results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 4, 5, 6], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 2
arg_2_2 = 3
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3_0_tensor = paddle.randint(-8,32,[1], dtype=paddle.int8)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
arg_4_0 = 3
arg_4_1 = 2
arg_4_2 = 4
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5_0 = -19
arg_5_1 = 89
arg_5_2 = 17
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
start = time.time()
results["time_low"] = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3_0 = arg_3_0_tensor.clone().type(paddle.int32)
arg_3 = [arg_3_0,]
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
start = time.time()
results["time_high"] = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
results["time_high"] = time.time() - start

print(results)
