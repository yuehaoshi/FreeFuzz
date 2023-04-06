import paddle
import numpy
import os

use_cuda = True
# NOTE: If you use CPU to run the program, you need
# to specify the CPU_NUM, otherwise, PaddlePaddle will use
# all the number of the logic core as the CPU_NUM,
# in that case, the batch size of the input should be
# greater than CPU_NUM, if not, the process will be
# failed by an exception.
if not use_cuda:
    os.environ['CPU_NUM'] = str(2)

paddle.enable_static()
train_program = paddle.static.Program()
startup_program = paddle.static.Program()
with paddle.static.program_guard(train_program, startup_program):
    data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
    hidden = paddle.static.nn.fc(data, 10)
    loss = paddle.mean(hidden)

place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(startup_program)

parallel_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
                                              main_program=train_program,
                                              loss_name=loss.name)

x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = parallel_exe.run(feed={"X": x},
                              fetch_list=[loss.name])

parallel_exe.drop_local_exe_scopes()