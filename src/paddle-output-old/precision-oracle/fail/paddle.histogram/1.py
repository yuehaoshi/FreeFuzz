results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,8,[3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 992
arg_3 = 0
arg_4 = 30
start = time.time()
results["time_low"] = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
results["time_high"] = time.time() - start

print(results)
