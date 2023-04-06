import paddle
import paddle.distributed as dist

# execute this command in terminal: export PADDLE_TRAINER_ID=0
print("The rank is %d" % dist.get_rank())
# The rank is 0