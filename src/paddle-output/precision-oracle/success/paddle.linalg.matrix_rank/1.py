results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([1, 7, 9, 3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.01
arg_3 = False
start = time.time()
results["time_low"] = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
start = time.time()
results["time_high"] = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
results["time_high"] = time.time() - start

print(results)
