results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[6, 2, 3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 0
arg_2_2 = 2
arg_2_3 = -1
arg_2_4 = 3
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
start = time.time()
results["time_low"] = paddle.sparse.reshape(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
start = time.time()
results["time_high"] = paddle.sparse.reshape(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
