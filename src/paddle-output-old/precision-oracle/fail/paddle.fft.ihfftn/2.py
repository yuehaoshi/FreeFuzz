results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,16,[5, 5], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -9
arg_2_1 = -34
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = -2
arg_3_1 = -1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = None
start = time.time()
results["time_low"] = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)
