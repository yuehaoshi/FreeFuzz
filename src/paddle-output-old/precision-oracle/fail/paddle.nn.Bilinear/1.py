results = dict()
import paddle
import time
arg_1 = 5
arg_2 = 26
arg_3 = "mean"
start = time.time()
results["time_low"] = paddle.nn.Bilinear(in1_features=arg_1,in2_features=arg_2,out_features=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.Bilinear(in1_features=arg_1,in2_features=arg_2,out_features=arg_3,)
results["time_high"] = time.time() - start

print(results)
