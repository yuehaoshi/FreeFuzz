results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,16,[7], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3_0 = 5
arg_3 = [arg_3_0,]
arg_4_0 = 6
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = paddle.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_3 = [arg_3_0,]
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = paddle.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
results["time_high"] = time.time() - start

print(results)
