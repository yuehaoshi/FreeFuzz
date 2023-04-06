# required: distributed
import numpy as np
import paddle
from paddle.distributed import init_parallel_env

init_parallel_env()
out_tensor_list = []
if paddle.distributed.ParallelEnv().rank == 0:
    np_data1 = np.array([[1, 2, 3], [4, 5, 6]])
    np_data2 = np.array([[7, 8, 9], [10, 11, 12]])
else:
    np_data1 = np.array([[13, 14, 15], [16, 17, 18]])
    np_data2 = np.array([[19, 20, 21], [22, 23, 24]])
data1 = paddle.to_tensor(np_data1)
data2 = paddle.to_tensor(np_data2)
paddle.distributed.alltoall([data1, data2], out_tensor_list)
# out for rank 0: [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]]
# out for rank 1: [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]]