results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([128, 32], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = True
start = time.time()
results["time_low"] = paddle.matmul(arg_1,arg_2,transpose_y=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.matmul(arg_1,arg_2,transpose_y=arg_3,)
results["time_high"] = time.time() - start

print(results)
