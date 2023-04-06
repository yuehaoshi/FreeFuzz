import paddle
import paddle.distributed as dist

# execute this command in terminal: export PADDLE_TRAINERS_NUM=4
print("The world_size is %d" % dist.get_world_size())
# The world_size is 4