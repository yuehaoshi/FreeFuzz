results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1024,32,[3, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
start = time.time()
results["time_low"] = paddle.linalg.matrix_power(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.linalg.matrix_power(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)