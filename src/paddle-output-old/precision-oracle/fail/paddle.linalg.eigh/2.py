results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2 = -47
start = time.time()
results["time_low"] = paddle.linalg.eigh(arg_1,UPLO=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
start = time.time()
results["time_high"] = paddle.linalg.eigh(arg_1,UPLO=arg_2,)
results["time_high"] = time.time() - start

print(results)
