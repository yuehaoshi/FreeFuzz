results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([2, 2], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([2, 2], dtype=paddle.float16)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2_tensor = paddle.randint(-128,32,[3, 2], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.multiplex(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.multiplex(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
