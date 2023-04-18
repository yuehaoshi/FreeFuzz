results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,4,[1, 2, 3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = 2
start = time.time()
results["time_low"] = paddle.flatten(arg_1,start_axis=arg_2,stop_axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.flatten(arg_1,start_axis=arg_2,stop_axis=arg_3,)
results["time_high"] = time.time() - start

print(results)
