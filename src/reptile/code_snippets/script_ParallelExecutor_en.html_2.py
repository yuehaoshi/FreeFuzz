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
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

exe.run(startup_program)

train_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
                                           main_program=train_program,
                                           loss_name=loss.name)

# If the feed is a dict:
# the image will be split into devices. If there is two devices
# each device will process an image with shape (5, 1)
x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = train_exe.run(feed={"X": x},
                           fetch_list=[loss.name])

# If the feed is a list:
# each device will process each element in the list.
# the 1st device will process an image with shape (10, 1)
# the 2nd device will process an image with shape (9, 1)
#
# you can use exe.device_count to get the device number.
x2 = numpy.random.random(size=(9, 1)).astype('float32')
loss_data, = train_exe.run(feed=[{"X": x}, {"X": x2}],
                           fetch_list=[loss.name])