results = dict()
import paddle
import time
arg_1 = "slot4"
arg_2_0 = None
arg_2_1 = 40
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "float32"
arg_4 = 1
start = time.time()
results["time_low"] = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,lod_level=arg_4,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,lod_level=arg_4,)
results["time_high"] = time.time() - start

print(results)
