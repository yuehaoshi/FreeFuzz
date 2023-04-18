results = dict()
import paddle
import time
arg_1 = 3
arg_2 = 2
arg_3 = "max"
arg_class = paddle.nn.AvgPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
