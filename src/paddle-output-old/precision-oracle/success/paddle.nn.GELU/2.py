results = dict()
import paddle
import time
arg_1 = True
start = time.time()
results["time_low"] = paddle.nn.GELU(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.GELU(arg_1,)
results["time_high"] = time.time() - start

print(results)
