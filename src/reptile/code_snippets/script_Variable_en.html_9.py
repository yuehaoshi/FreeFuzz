import paddle
paddle.enable_static()

x = paddle.static.data(name='x1', shape=[3, 2], dtype='bool')
x.element_size() # 1

x = paddle.static.data(name='x2', shape=[3, 2], dtype='int16')
x.element_size() # 2

x = paddle.static.data(name='x3', shape=[3, 2], dtype='float16')
x.element_size() # 2

x = paddle.static.data(name='x4', shape=[3, 2], dtype='float32')
x.element_size() # 4

x = paddle.static.data(name='x5', shape=[3, 2], dtype='float64')
x.element_size() # 8