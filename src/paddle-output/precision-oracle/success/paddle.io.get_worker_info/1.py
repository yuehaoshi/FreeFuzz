results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.io.get_worker_info()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.io.get_worker_info()
results["time_high"] = time.time() - start

print(results)