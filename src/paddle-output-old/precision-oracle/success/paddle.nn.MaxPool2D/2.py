results = dict()
import paddle
import time
arg_1 = 2
arg_2 = 2
start = time.time()
results["time_low"] = paddle.nn.MaxPool2D(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.MaxPool2D(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)