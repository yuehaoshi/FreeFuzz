# execute this command in terminal: export PADDLE_CURRENT_ENDPOINT=127.0.0.1:6170
import paddle.distributed as dist

env = dist.ParallelEnv()
print("The current endpoint are %s" % env.current_endpoint)
# The current endpoint are 127.0.0.1:6170