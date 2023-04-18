results = dict()
import paddle
import time
arg_1 = -60
arg_2 = 4
arg_3 = 1000
arg_class = paddle.nn.Bilinear(in1_features=arg_1,in2_features=arg_2,out_features=arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
