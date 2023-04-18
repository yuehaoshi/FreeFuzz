results = dict()
import paddle
import time
arg_1 = 1
arg_2 = False
arg_3 = 3
arg_4 = True
arg_5 = 1
arg_class = paddle.nn.Conv2D(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,)
arg_6 = None
start = time.time()
results["time_low"] = arg_class(*arg_6)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_6)
results["time_high"] = time.time() - start

print(results)
