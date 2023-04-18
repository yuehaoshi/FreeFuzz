results = dict()
import paddle
import time
arg_1 = -12
arg_2 = 2
arg_3 = 0
start = time.time()
results["time_low"] = paddle.nn.AvgPool3D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.AvgPool3D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
results["time_high"] = time.time() - start

print(results)
