results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = "int32"
arg_3 = "float64"
start = time.time()
results["time_low"] = paddle.sparse.cast(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
start = time.time()
results["time_high"] = paddle.sparse.cast(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)