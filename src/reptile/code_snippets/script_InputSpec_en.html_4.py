from paddle.static import InputSpec

x_spec = InputSpec(shape=[64], dtype='float32', name='x')
x_spec.batch(4)
print(x_spec) # InputSpec(shape=(4, 64), dtype=paddle.float32, name=x)