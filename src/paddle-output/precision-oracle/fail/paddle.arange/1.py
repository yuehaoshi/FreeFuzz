results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3 = -93.0
start = time.time()
results["time_low"] = paddle.arange(arg_1,arg_2,dtype=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int32)
arg_2 = arg_2_tensor.clone().astype(paddle.int32)
start = time.time()
results["time_high"] = paddle.arange(arg_1,arg_2,dtype=arg_3,)
results["time_high"] = time.time() - start

print(results)
