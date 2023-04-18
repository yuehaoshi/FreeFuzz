results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,2,[4], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,8,[2, 3, 2], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-4,16384,[4], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.where(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.bool)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.where(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
