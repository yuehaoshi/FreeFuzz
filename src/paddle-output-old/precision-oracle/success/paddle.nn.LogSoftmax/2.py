results = dict()
import paddle
import time
arg_1 = "max"
start = time.time()
results["time_low"] = paddle.nn.LogSoftmax(axis=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.LogSoftmax(axis=arg_1,)
results["time_high"] = time.time() - start

print(results)
