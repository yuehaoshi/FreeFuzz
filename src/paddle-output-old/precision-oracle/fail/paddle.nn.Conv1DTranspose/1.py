results = dict()
import paddle
import time
arg_1 = 2
arg_2 = "max"
arg_3 = 64
start = time.time()
results["time_low"] = paddle.nn.Conv1DTranspose(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.Conv1DTranspose(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
