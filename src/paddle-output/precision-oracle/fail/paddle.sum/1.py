results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = False
arg_4 = "equal_nan"
start = time.time()
results["time_low"] = paddle.sum(arg_1,arg_2,keepdim=arg_3,name=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.sum(arg_1,arg_2,keepdim=arg_3,name=arg_4,)
results["time_high"] = time.time() - start

print(results)
