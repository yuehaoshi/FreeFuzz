results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[3, 2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 9, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 1024
arg_3_1 = 58
arg_3_2 = -21
arg_3_3 = 16
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
start = time.time()
results["time_low"] = paddle.scatter_nd(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
arg_2 = arg_2_tensor.clone().astype(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
start = time.time()
results["time_high"] = paddle.scatter_nd(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
