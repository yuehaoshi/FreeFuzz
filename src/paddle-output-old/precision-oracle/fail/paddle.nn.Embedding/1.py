results = dict()
import paddle
import time
arg_1 = 45
arg_2 = False
start = time.time()
results["time_low"] = paddle.nn.Embedding(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.Embedding(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
