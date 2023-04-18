results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = True
arg_2_1 = -20.0
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.static.append_backward(loss=arg_1,parameter_list=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.static.append_backward(loss=arg_1,parameter_list=arg_2,)
results["time_high"] = time.time() - start

print(results)
