results = dict()
import paddle
import time
arg_1 = False
start = time.time()
results["time_low"] = paddle.nn.InstanceNorm2D(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.InstanceNorm2D(arg_1,)
results["time_high"] = time.time() - start

print(results)