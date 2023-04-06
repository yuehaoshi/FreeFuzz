import paddle
import paddle.static as static

paddle.enable_static()

exec_strategy = static.ExecutionStrategy()
exec_strategy.num_iteration_per_run = 10