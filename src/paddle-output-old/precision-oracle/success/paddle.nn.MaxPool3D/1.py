results = dict()
import paddle
import time
arg_1 = "max"
arg_2 = -1074
arg_3 = True
start = time.time()
results["time_low"] = paddle.nn.MaxPool3D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.MaxPool3D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
results["time_high"] = time.time() - start

print(results)
