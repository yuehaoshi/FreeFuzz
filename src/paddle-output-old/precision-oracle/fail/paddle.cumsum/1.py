results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,4,[3, 4], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = "float64"
start = time.time()
results["time_low"] = paddle.cumsum(arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.cumsum(arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)
