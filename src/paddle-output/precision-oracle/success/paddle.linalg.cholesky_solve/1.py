results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = True
start = time.time()
results["time_low"] = paddle.linalg.cholesky_solve(arg_1,arg_2,upper=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
arg_2 = arg_2_tensor.clone().astype(paddle.float64)
start = time.time()
results["time_high"] = paddle.linalg.cholesky_solve(arg_1,arg_2,upper=arg_3,)
results["time_high"] = time.time() - start

print(results)