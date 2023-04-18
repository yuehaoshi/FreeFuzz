results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-128,2,[3, 9, 5], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = True
arg_3 = 1
start = time.time()
results["time_low"] = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)
