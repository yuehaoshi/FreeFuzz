results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4,64,[2, 6], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,4,[2, 4], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.searchsorted(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.searchsorted(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)