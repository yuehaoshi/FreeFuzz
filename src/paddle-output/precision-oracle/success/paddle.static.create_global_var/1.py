results = dict()
import paddle
import time
arg_1_0 = 84
arg_1_1 = 46
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 1.0
arg_3 = "float32"
arg_4 = False
arg_5 = True
arg_6 = "new_var"
start = time.time()
results["time_low"] = paddle.static.create_global_var(shape=arg_1,value=arg_2,dtype=arg_3,persistable=arg_4,force_cpu=arg_5,name=arg_6,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.static.create_global_var(shape=arg_1,value=arg_2,dtype=arg_3,persistable=arg_4,force_cpu=arg_5,name=arg_6,)
results["time_high"] = time.time() - start

print(results)
