results = dict()
import paddle
import time
arg_1 = "func_not_to_static"
start = time.time()
results["time_low"] = paddle.jit.not_to_static(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.jit.not_to_static(arg_1,)
results["time_high"] = time.time() - start

print(results)
