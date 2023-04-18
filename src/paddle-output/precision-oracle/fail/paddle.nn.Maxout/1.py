results = dict()
import paddle
import time
arg_1 = -16
arg_class = paddle.nn.Maxout(groups=arg_1,)
arg_2 = True
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
