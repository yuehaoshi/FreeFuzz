results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = "int64"
start = time.time()
results["time_low"] = paddle.ones(arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int32)
start = time.time()
results["time_high"] = paddle.ones(arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)
