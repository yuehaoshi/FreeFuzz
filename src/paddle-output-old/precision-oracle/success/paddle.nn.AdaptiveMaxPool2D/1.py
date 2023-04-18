results = dict()
import paddle
import time
arg_1 = 3
arg_2 = True
start = time.time()
results["time_low"] = paddle.nn.AdaptiveMaxPool2D(output_size=arg_1,return_mask=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.AdaptiveMaxPool2D(output_size=arg_1,return_mask=arg_2,)
results["time_high"] = time.time() - start

print(results)
