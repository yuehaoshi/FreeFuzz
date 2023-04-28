results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([10, 2, 5], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_0_tensor = paddle.rand([10, 2, 5], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3 = None
arg_4 = True
arg_5 = "max"
start = time.time()
results["time_low"] = paddle.grad(arg_1,arg_2,arg_3,create_graph=arg_4,allow_unused=arg_5,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.float32)
arg_1 = [arg_1_0,]
arg_2_0 = arg_2_0_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.grad(arg_1,arg_2,arg_3,create_graph=arg_4,allow_unused=arg_5,)
results["time_high"] = time.time() - start

print(results)
