results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4,2,[2, 2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = 38
start = time.time()
results["time_low"] = paddle.diag(arg_1,padding_value=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.diag(arg_1,padding_value=arg_2,)
results["time_high"] = time.time() - start

print(results)