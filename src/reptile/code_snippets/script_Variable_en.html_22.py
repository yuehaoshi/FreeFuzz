import paddle

paddle.enable_static()

# create a static Variable
x = paddle.static.data(name='x', shape=[3, 2, 1])

# get the number of elements of the Variable
y = x.size()