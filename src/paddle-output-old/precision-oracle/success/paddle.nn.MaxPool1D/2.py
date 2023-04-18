results = dict()
import paddle
import time
arg_1 = -42
arg_2 = -1
arg_3 = "max"
start = time.time()
results["time_low"] = paddle.nn.MaxPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.MaxPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
results["time_high"] = time.time() - start

print(results)
