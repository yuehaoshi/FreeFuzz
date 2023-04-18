results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([-1, 10], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,32,[-1, 1], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 5
start = time.time()
results["time_low"] = paddle.static.accuracy(input=arg_1,label=arg_2,k=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.static.accuracy(input=arg_1,label=arg_2,k=arg_3,)
results["time_high"] = time.time() - start

print(results)
