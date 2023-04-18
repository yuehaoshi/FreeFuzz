results = dict()
import paddle
import time
arg_1 = -1
arg_2 = -48
arg_3 = False
start = time.time()
results["time_low"] = paddle.nn.AvgPool2D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.AvgPool2D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
results["time_high"] = time.time() - start

print(results)
