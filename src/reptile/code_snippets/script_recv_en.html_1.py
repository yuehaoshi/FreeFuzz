# required: distributed
import paddle
from paddle.distributed import init_parallel_env

init_parallel_env()
if paddle.distributed.ParallelEnv().rank == 0:
    data = paddle.to_tensor([7, 8, 9])
    paddle.distributed.send(data, dst=1)
else:
    data = paddle.to_tensor([1,2,3])
    paddle.distributed.recv(data, src=0)
out = data.numpy()