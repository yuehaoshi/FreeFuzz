import paddle

paddle.enable_static()

# create a static Variable
x = paddle.static.data(name='x', shape=[3, 2, 1])
# create a cloned Variable
y = x.clone()