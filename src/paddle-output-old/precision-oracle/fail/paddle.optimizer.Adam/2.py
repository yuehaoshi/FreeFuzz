results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.optimizer.Adam()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.optimizer.Adam()
results["time_high"] = time.time() - start

print(results)
