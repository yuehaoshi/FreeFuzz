results = dict()
import paddle
import time
arg_1 = 0.001
start = time.time()
results["time_low"] = paddle.optimizer.Adam(learning_rate=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.optimizer.Adam(learning_rate=arg_1,)
results["time_high"] = time.time() - start

print(results)
