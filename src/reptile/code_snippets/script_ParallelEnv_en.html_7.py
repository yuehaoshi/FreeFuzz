# execute this command in terminal: export FLAGS_nccl_nrings=1
import paddle.distributed as dist

env = dist.ParallelEnv()
print("The nrings is %d" % env.nrings)
# the number of ring is 1