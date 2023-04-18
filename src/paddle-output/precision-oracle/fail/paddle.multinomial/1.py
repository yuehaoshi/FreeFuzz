results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([52, 4, 1], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
start = time.time()
results["time_low"] = paddle.multinomial(arg_1,num_samples=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.complex128)
start = time.time()
results["time_high"] = paddle.multinomial(arg_1,num_samples=arg_2,)
results["time_high"] = time.time() - start

print(results)
