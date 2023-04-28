results = dict()
import paddle
import time
arg_1 = "element_wise_add"
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_0_tensor = int8_tensor
arg_2_0 = arg_2_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2, 3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_1_tensor = int8_tensor
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
int_tensor = paddle.randint(low=-128, high=127, shape=[3, 1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_tensor = int8_tensor
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().astype(paddle.int32)
arg_2_1 = arg_2_1_tensor.clone().astype(paddle.int32)
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = arg_3_tensor.clone().astype(paddle.int32)
start = time.time()
results["time_high"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
results["time_high"] = time.time() - start

print(results)
