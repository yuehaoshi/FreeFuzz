results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([1, 2, 3, 4], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([1, 2, 1, 4], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.rand([1, 1, 3, 1], dtype=paddle.float32)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2_tensor = paddle.randint(-4,4,[-1, 1], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 5
start = time.time()
results["time_low"] = paddle.static.accuracy(input=arg_1,label=arg_2,k=arg_3,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1_2 = arg_1_2_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = arg_2_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.static.accuracy(input=arg_1,label=arg_2,k=arg_3,)
results["time_high"] = time.time() - start

print(results)
