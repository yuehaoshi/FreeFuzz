results = dict()
import paddle
import time
arg_1 = 37.0
arg_class = paddle.nn.AdaptiveAvgPool2D(output_size=arg_1,)
arg_2 = None
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
