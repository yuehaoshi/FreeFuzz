import numpy as np
import paddle
from paddle.distributed import init_parallel_env

# required: gpu

paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()
if paddle.distributed.ParallelEnv().local_rank == 0:
    np_data1 = np.array([7, 8, 9])
    np_data2 = np.array([10, 11, 12])
else:
    np_data1 = np.array([1, 2, 3])
    np_data2 = np.array([4, 5, 6])
data1 = paddle.to_tensor(np_data1)
data2 = paddle.to_tensor(np_data2)
if paddle.distributed.ParallelEnv().local_rank == 0:
    paddle.distributed.scatter(data1, src=1)
else:
    paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
out = data1.numpy()