results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,4,[4, 1], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -36.0
arg_2_1 = 25.0
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.tile(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.tile(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
