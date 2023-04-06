# execute this command in terminal: export PADDLE_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171
import paddle.distributed as dist

env = dist.ParallelEnv()
print("The trainer endpoints are %s" % env.trainer_endpoints)
# The trainer endpoints are ['127.0.0.1:6170', '127.0.0.1:6171']