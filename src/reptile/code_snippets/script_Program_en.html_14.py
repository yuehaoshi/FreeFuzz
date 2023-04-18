import paddle
import paddle.static as static

paddle.enable_static()

prog = static.default_main_program()
img = static.data(name='img', shape=[None, 1,28,28], dtype='float32')
label = static.data(name='label', shape=[None,1], dtype='int64')
for var in prog.list_vars():
    print(var)

# var img : LOD_TENSOR.shape(-1, 1, 28, 28).dtype(float32).stop_gradient(True)
# var label : LOD_TENSOR.shape(-1, 1).dtype(int64).stop_gradient(True)