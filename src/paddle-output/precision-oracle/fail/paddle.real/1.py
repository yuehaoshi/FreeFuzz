results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32,4,[3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.real(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.real(arg_1,)
results["time_high"] = time.time() - start

print(results)
