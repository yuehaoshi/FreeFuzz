results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16,2,[2, 2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.round(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.round(arg_1,)
results["time_high"] = time.time() - start

print(results)
