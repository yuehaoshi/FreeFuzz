import paddle
import numpy
import os

use_cuda = True
paddle.enable_static()
place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

# NOTE: If you use CPU to run the program, you need
# to specify the CPU_NUM, otherwise, PaddlePaddle will use
# all the number of the logic core as the CPU_NUM,
# in that case, the batch size of the input should be
# greater than CPU_NUM, if not, the process will be
# failed by an exception.
if not use_cuda:
    os.environ['CPU_NUM'] = str(2)

exe = paddle.static.Executor(place)

train_program = paddle.static.Program()
startup_program = paddle.static.Program()
with paddle.static.program_guard(train_program, startup_program):
    data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
    hidden = paddle.static.nn.fc(data, 10)
    loss = paddle.mean(hidden)
    test_program = paddle.static.default_main_program().clone(for_test=True)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

exe.run(startup_program)

train_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
                                           main_program=train_program,
                                           loss_name=loss.name)
# Note: if share_vars_from is not set here, the test parameter is different to the train one
test_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
                                          main_program=test_program,
                                          share_vars_from=train_exe)

x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = train_exe.run(feed={"X": x},
                           fetch_list=[loss.name])

loss_data, = test_exe.run(feed={"X": x},
                          fetch_list=[loss.name])