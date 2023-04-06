# execute this command in terminal: export FLAGS_selected_gpus=1
import paddle.distributed as dist

env = dist.ParallelEnv()
print("The device id are %d" % env.device_id)
# The device id are 1