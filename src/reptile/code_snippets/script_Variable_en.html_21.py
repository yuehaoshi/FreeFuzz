import paddle
import paddle.static as static
import numpy as np

paddle.enable_static()

x = static.data(name="x", shape=[10, 10], dtype='float32')

y = static.nn.fc(x, 10, name='fc')
place = paddle.CPUPlace()
exe = static.Executor(place)
prog = paddle.static.default_main_program()
exe.run(static.default_startup_program())
inputs = np.ones((10, 10), dtype='float32')
exe.run(prog, feed={'x': inputs}, fetch_list=[y, ])
path = 'temp/tensor_'
for var in prog.list_vars():
    if var.persistable:
        t = var.get_value()
        paddle.save(t, path+var.name+'.pdtensor')

for var in prog.list_vars():
    if var.persistable:
        t_load = paddle.load(path+var.name+'.pdtensor')
        var.set_value(t_load)