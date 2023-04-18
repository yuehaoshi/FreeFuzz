results = dict()
import paddle
import time
arg_1 = 68
arg_2 = 32
arg_3 = -17
start = time.time()
results["time_low"] = paddle.nn.SimpleRNN(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.SimpleRNN(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
