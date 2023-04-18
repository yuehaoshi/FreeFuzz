results = dict()
import paddle
import time
arg_1 = 6
arg_2 = 16
arg_3 = 35
arg_4 = "sum"
arg_5 = 0
start = time.time()
results["time_low"] = paddle.nn.Conv2D(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.Conv2D(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,)
results["time_high"] = time.time() - start

print(results)
