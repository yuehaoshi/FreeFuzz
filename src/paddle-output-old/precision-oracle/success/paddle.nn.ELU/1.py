results = dict()
import paddle
import time
arg_1 = 5.0
start = time.time()
results["time_low"] = paddle.nn.ELU(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.ELU(arg_1,)
results["time_high"] = time.time() - start

print(results)
