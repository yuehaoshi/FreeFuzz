results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[1, 2, 3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[3])
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.minimum(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
arg_2 = arg_2_tensor.clone().astype(paddle.bool)
start = time.time()
results["time_high"] = paddle.minimum(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
