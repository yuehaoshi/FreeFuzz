results = dict()
import paddle
import time
arg_1 = -95.0
arg_2 = 1
arg_3 = 1024
arg_class = paddle.nn.Conv1DTranspose(arg_1,arg_2,arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
