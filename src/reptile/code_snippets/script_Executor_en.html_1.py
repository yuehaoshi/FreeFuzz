import paddle
import numpy
import os

# Executor is only used in static graph mode
paddle.enable_static()

# Set place explicitly.
# use_cuda = True
# place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
# exe = paddle.static.Executor(place)

# If you don't set place, PaddlePaddle sets the default device.
exe = paddle.static.Executor()

train_program = paddle.static.Program()
startup_program = paddle.static.Program()
with paddle.static.program_guard(train_program, startup_program):
    data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
    hidden = paddle.static.nn.fc(data, 10)
    loss = paddle.mean(hidden)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# Run the startup program once and only once.
# Not need to optimize/compile the startup program.
exe.run(startup_program)

# Run the main program directly without compile.
x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(train_program, feed={"X": x}, fetch_list=[loss.name])

# Or, compiled the program and run. See `CompiledProgram`
# for more details.
# NOTE: If you use CPU to run the program or Paddle is
# CPU version, you need to specify the CPU_NUM, otherwise,
# PaddlePaddle will use all the number of the logic core as
# the CPU_NUM, in that case, the batch size of the input
# should be greater than CPU_NUM, if not, the process will be
# failed by an exception.

# Set place explicitly.
# if not use_cuda:
#     os.environ['CPU_NUM'] = str(2)

# If you don't set place and PaddlePaddle is CPU version
os.environ['CPU_NUM'] = str(2)

compiled_prog = paddle.static.CompiledProgram(
    train_program).with_data_parallel(loss_name=loss.name)
loss_data, = exe.run(compiled_prog, feed={"X": x}, fetch_list=[loss.name])