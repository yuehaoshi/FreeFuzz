results = dict()
import paddle
import time
arg_1 = 16
arg_2 = -10
arg_3 = 2
start = time.time()
results["time_low"] = paddle.nn.SimpleRNN(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.SimpleRNN(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)