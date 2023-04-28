results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[6, 7, 8], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 0
arg_3 = [arg_3_0,]
arg_4_0 = -105.0
arg_4_1 = -1e+20
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = -2
arg_5_1 = -3
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_low"] = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,]
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = [arg_5_0,arg_5_1,]
start = time.time()
results["time_high"] = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
results["time_high"] = time.time() - start

print(results)
