# execute this command in terminal: export PADDLE_TRAINER_ID=0
import paddle.distributed as dist

env = dist.ParallelEnv()
print("The rank is %d" % env.rank)
# The rank is 0