results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-4,8,[10, 1, 5, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,8,[1, 3, 2, 5], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.matmul(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.matmul(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
