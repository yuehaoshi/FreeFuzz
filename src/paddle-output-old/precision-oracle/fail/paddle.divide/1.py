results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = None
start = time.time()
results["time_low"] = paddle.divide(arg_1,arg_2,name=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.divide(arg_1,arg_2,name=arg_3,)
results["time_high"] = time.time() - start

print(results)
