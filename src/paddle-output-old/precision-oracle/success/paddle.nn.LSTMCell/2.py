results = dict()
import paddle
import time
arg_1 = 16
arg_2 = 32
start = time.time()
results["time_low"] = paddle.nn.LSTMCell(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.LSTMCell(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
