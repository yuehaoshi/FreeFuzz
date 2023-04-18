results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,64,[0, 2, 1], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -2
arg_3_1 = -1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = True
arg_5 = None
start = time.time()
results["time_low"] = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)
