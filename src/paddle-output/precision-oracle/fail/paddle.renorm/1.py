results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = 12
arg_4 = 63.0
start = time.time()
results["time_low"] = paddle.renorm(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.renorm(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
