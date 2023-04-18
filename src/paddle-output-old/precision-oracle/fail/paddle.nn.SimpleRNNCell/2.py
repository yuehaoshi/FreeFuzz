results = dict()
import paddle
import time
arg_1 = "zeros"
arg_2 = 32
start = time.time()
results["time_low"] = paddle.nn.SimpleRNNCell(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.SimpleRNNCell(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
