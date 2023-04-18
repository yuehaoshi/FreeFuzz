results = dict()
import paddle
import time
arg_class = paddle.nn.Hardsigmoid()
arg_1 = None
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)
