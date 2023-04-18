results = dict()
import paddle
import time
arg_1 = 4
start = time.time()
results["time_low"] = paddle.nn.AdaptiveMaxPool1D(output_size=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.AdaptiveMaxPool1D(output_size=arg_1,)
results["time_high"] = time.time() - start

print(results)
