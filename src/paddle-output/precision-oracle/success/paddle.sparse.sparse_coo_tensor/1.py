results = dict()
import paddle
import time
arg_1_0_0 = 0
arg_1_0_1 = 0
arg_1_0_2 = 1
arg_1_0 = [arg_1_0_0,arg_1_0_1,arg_1_0_2,]
arg_1_1_0 = 1
arg_1_1_1 = 1
arg_1_1_2 = 2
arg_1_1 = [arg_1_1_0,arg_1_1_1,arg_1_1_2,]
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0 = 8.0
arg_2_1 = 13.0
arg_2_2 = -17.0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = [arg_1_0_0,arg_1_0_1,arg_1_0_2,]
arg_1_1 = [arg_1_1_0,arg_1_1_1,arg_1_1_2,]
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.sparse.sparse_coo_tensor(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
