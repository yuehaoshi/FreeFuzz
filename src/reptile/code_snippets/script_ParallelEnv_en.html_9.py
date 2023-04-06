# execute this command in terminal: export PADDLE_TRAINERS_NUM=4
import paddle.distributed as dist

env = dist.ParallelEnv()
print("The world_size is %d" % env.world_size)
# The world_size is 4