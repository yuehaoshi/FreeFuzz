results = dict()
import paddle
import time
arg_1 = -32768
arg_2 = 4096
arg_3_0 = 3
arg_3_1 = 4
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "paddleVarType"
start = time.time()
results["time_low"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_high"] = time.time() - start

print(results)
