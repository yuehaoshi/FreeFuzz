import paddle
import numpy

# First create the Executor.
paddle.enable_static()
place = paddle.CPUPlace()  # paddle.CUDAPlace(0)
exe = paddle.static.Executor(place)

data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
hidden = paddle.static.nn.fc(data, 10)
loss = paddle.mean(hidden)
adam = paddle.optimizer.Adam()
adam.minimize(loss)
i = paddle.zeros(shape=[1], dtype='int64')
array = paddle.fluid.layers.array_write(x=loss, i=i)

# Run the startup program once and only once.
exe.run(paddle.static.default_startup_program())

x = numpy.random.random(size=(10, 1)).astype('float32')
loss_val, array_val = exe.run(feed={'X': x},
                              fetch_list=[loss.name, array.name])
print(array_val)
# [array([0.02153828], dtype=float32)]