results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,64,[3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,128,[2, 3, 2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.bitwise_and(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.bitwise_and(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)