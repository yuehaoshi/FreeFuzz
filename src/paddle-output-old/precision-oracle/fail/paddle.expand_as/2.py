results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8192,32,[8, 48000], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,64,[2, 3], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.expand_as(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.expand_as(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
