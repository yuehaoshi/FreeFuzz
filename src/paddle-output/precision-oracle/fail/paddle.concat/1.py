results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([-1, 608, 14, 14], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([-1, 32, 14, 14], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 1
start = time.time()
results["time_low"] = paddle.concat(arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().astype(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.concat(arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)
