results = dict()
import paddle
import time
arg_1 = 79.0
arg_2_0_0 = 1
arg_2_0_1 = 1
arg_2_0_2 = 28
arg_2_0_3 = 28
arg_2_0 = [arg_2_0_0,arg_2_0_1,arg_2_0_2,arg_2_0_3,]
arg_2_1_0 = 1
arg_2_1_1 = 400
arg_2_1 = [arg_2_1_0,arg_2_1_1,]
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = "float32"
arg_3_1 = "float32"
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.summary(arg_1,arg_2,dtypes=arg_3,)
results["time_low"] = time.time() - start
arg_2_0 = [arg_2_0_0,arg_2_0_1,arg_2_0_2,arg_2_0_3,]
arg_2_1 = [arg_2_1_0,arg_2_1_1,]
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.summary(arg_1,arg_2,dtypes=arg_3,)
results["time_high"] = time.time() - start

print(results)
