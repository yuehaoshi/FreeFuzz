results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(0,2,[2, 1], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3, 4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = -21
arg_3_1 = 42
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.bool)
arg_2 = arg_2_tensor.clone().type(paddle.float64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
results["time_high"] = time.time() - start

print(results)
