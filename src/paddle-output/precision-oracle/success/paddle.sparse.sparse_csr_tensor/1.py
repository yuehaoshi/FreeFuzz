results = dict()
import paddle
import time
arg_1_0 = 0
arg_1_1 = 2
arg_1_2 = 3
arg_1_3 = 5
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2_0 = 1
arg_2_1 = 3
arg_2_2 = 2
arg_2_3 = 0
arg_2_4 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
arg_3_0 = 1.0
arg_3_1 = 2.0
arg_3_2 = 3.0
arg_3_3 = 4.0
arg_3_4 = 5.0
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,]
arg_4_0 = 3
arg_4_1 = 4
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_low"] = paddle.sparse.sparse_csr_tensor(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,]
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_high"] = paddle.sparse.sparse_csr_tensor(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
