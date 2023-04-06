import numpy as np
import paddle
paddle.enable_static()

# Creates a variable with fixed size [3, 2, 1]
# User can only feed data of the same shape to x
# the dtype is not set, so it will set "float32" by
# paddle.get_default_dtype(). You can use paddle.get_default_dtype() to
# change the global dtype
x = paddle.static.data(name='x', shape=[3, 2, 1])

# Creates a variable with changeable batch size -1.
# Users can feed data of any batch size into y,
# but size of each data sample has to be [2, 1]
y = paddle.static.data(name='y', shape=[-1, 2, 1], dtype='float32')

z = x + y

# In this example, we will feed x and y with np-ndarray "1"
# and fetch z, like implementing "1 + 1 = 2" in PaddlePaddle
feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

exe = paddle.static.Executor(paddle.framework.CPUPlace())
out = exe.run(paddle.static.default_main_program(),
              feed={
                  'x': feed_data,
                  'y': feed_data
              },
              fetch_list=[z.name])

# np-ndarray of shape=[3, 2, 1], dtype=float32, whose elements are 2
print(out)