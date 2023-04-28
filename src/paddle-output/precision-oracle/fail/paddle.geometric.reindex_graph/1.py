results = dict()
import paddle
import time
int_tensor = paddle.randint(low=-128, high=127, shape=[3], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[7], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_tensor = int8_tensor
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([18], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = paddle.geometric.reindex_graph(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.int64)
arg_2 = arg_2_tensor.clone().astype(paddle.int64)
arg_3 = arg_3_tensor.clone().astype(paddle.float16)
start = time.time()
results["time_high"] = paddle.geometric.reindex_graph(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
