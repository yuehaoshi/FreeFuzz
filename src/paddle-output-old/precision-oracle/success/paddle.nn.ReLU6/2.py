results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.nn.ReLU6()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.ReLU6()
results["time_high"] = time.time() - start

print(results)
