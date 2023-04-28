results = dict()
import paddle
import time
arg_1 = -14.0
start = time.time()
results["time_low"] = paddle.static.cpu_places(device_count=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.static.cpu_places(device_count=arg_1,)
results["time_high"] = time.time() - start

print(results)
