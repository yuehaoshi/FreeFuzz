results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 2, 2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = -16
arg_3_0 = 1
arg_3_1 = 2
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.rot90(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.rot90(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
