results = dict()
import paddle
import time
start = time.time()
results["time_low"] = paddle.nn.Softsign()
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.Softsign()
results["time_high"] = time.time() - start

print(results)
