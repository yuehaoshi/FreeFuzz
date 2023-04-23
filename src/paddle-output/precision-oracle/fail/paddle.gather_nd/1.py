results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,64,[0, 3, 2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,128,[1, 2], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.gather_nd(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.gather_nd(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)