results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,2,[5, 1])
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[5, 0])
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.logical_and(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.bool)
arg_2 = arg_2_tensor.clone().astype(paddle.bool)
start = time.time()
results["time_high"] = paddle.logical_and(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
