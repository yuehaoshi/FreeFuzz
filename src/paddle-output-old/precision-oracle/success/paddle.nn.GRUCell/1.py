results = dict()
import paddle
import time
arg_1 = 32
arg_2 = 32
start = time.time()
results["time_low"] = paddle.nn.GRUCell(input_size=arg_1,hidden_size=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.GRUCell(input_size=arg_1,hidden_size=arg_2,)
results["time_high"] = time.time() - start

print(results)