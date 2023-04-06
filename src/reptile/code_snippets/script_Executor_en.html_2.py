import paddle

cpu = paddle.CPUPlace()
exe = paddle.static.Executor(cpu)
# execute training or testing
exe.close()