import numpy
import paddle
import paddle.static as static
from paddle.static import ExponentialMovingAverage

paddle.enable_static()

data = static.data(name='x', shape=[-1, 5], dtype='float32')
hidden = static.nn.fc(x=data, size=10)
cost = paddle.mean(hidden)

test_program = static.default_main_program().clone(for_test=True)
optimizer = paddle.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(cost)

ema = ExponentialMovingAverage(0.999)
ema.update()

place = paddle.CPUPlace()
exe = static.Executor(place)
exe.run(static.default_startup_program())

for pass_id in range(3):
    for batch_id in range(6):
        data = numpy.random.random(size=(10, 5)).astype('float32')
        exe.run(program=static.default_main_program(),
        feed={'x': data},
        fetch_list=[cost.name])

    # usage 1
    with ema.apply(exe):
        data = numpy.random.random(size=(10, 5)).astype('float32')
        exe.run(program=test_program,
            feed={'x': data},
            fetch_list=[hidden.name])

    # usage 2
    with ema.apply(exe, need_restore=False):
        data = numpy.random.random(size=(10, 5)).astype('float32')
        exe.run(program=test_program,
            feed={'x': data},
            fetch_list=[hidden.name])
    ema.restore(exe)