results = dict()
import paddle
import time
arg_1 = True
arg_2_tensor = paddle.randint(-64,64,[2], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.io.random_split(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.io.random_split(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
