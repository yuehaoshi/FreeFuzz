results = dict()
import paddle
import time
arg_1 = "max"
arg_2 = -1135
arg_3 = True
arg_class = paddle.nn.MaxPool3D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
