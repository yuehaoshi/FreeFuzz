results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 55, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
arg_3 = 9
start = time.time()
results["time_low"] = paddle.clip(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.clip(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
