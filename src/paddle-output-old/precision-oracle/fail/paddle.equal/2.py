results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,128,[3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 1], dtype=paddle.complex64)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.equal(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.complex64)
start = time.time()
results["time_high"] = paddle.equal(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
