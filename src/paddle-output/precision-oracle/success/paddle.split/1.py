results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 4
arg_3 = -1
start = time.time()
results["time_low"] = paddle.split(arg_1,num_or_sections=arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.split(arg_1,num_or_sections=arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)
